"""Power-extent exporter.

Writes ``power_extent.geojson`` (per ADR-0024, amended) as the concave hull
of a case's power-system asset locations. The concave hull supersedes the
original convex-hull/bus-bounding-polygon rule because SMART-DS feeder sets
are geographically dispersed — the SFO convex hull spans Pacific-to-Central-
Valley with most of the interior empty.

This module is intentionally narrow: it depends only on numpy + shapely +
file I/O. SMART-DS / OpenDSS parsing helpers are added incrementally as
later TDD cycles need them.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from pyproj import Transformer
from scipy.cluster.vq import kmeans2
from shapely.geometry import MultiPoint, Polygon, box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

_ASSET_COORD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("lon", "lat"),
    ("location_lon", "location_lat"),
    ("from_lon", "from_lat"),
    ("to_lon", "to_lat"),
)


def concave_power_extent(
    points: Iterable[tuple[float, float]],
    *,
    alpha_ratio: float,
) -> BaseGeometry:
    """Return the concave hull (alpha shape) of ``points``.

    Parameters
    ----------
    points : iterable of (lon, lat)
    alpha_ratio : float in (0, 1]
        Passed to ``shapely.concave_hull``. Smaller values produce a tighter
        hull that follows clusters more closely; larger values approach the
        convex hull. ``0.05–0.1`` is the working range for SMART-DS feeder
        clusters per docs/issues/0030.
    """
    if not (0.0 < alpha_ratio <= 1.0):
        raise ValueError(
            f"alpha_ratio must be in (0, 1]; got {alpha_ratio!r}"
        )

    coords = list(points)
    if len(coords) < 3:
        raise ValueError(
            f"need at least 3 points to form a hull; got {len(coords)}"
        )

    multipoint = MultiPoint(coords)
    return shapely.concave_hull(multipoint, ratio=alpha_ratio)


def iter_buscoords(smart_ds_year_root: Path) -> Iterator[tuple[float, float]]:
    """Yield (lon, lat) from every ``Buscoords.dss`` under a SMART-DS year root.

    Buscoords.dss is space-separated ``bus_name lon lat``. Lines starting
    with ``!`` or ``//`` are OpenDSS comments; blank lines are ignored.
    """
    year_root = Path(smart_ds_year_root)
    for path in year_root.rglob("Buscoords.dss"):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("!", "//")):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                yield float(parts[1]), float(parts[2])
            except ValueError:
                continue


def write_smart_ds_power_extent(
    *,
    region_id: str,
    smart_ds_year_root: Path,
    output_path: Path,
    alpha_ratio: float,
) -> dict:
    """Walk Buscoords under ``smart_ds_year_root`` and write a concave-hull
    ``power_extent.geojson`` at ``output_path``. Returns a manifest dict
    suitable for indexing and audit.
    """
    points = list(iter_buscoords(smart_ds_year_root))
    if not points:
        raise FileNotFoundError(
            f"no Buscoords.dss rows found under {smart_ds_year_root}"
        )

    hull = concave_power_extent(points, alpha_ratio=alpha_ratio)
    convex = MultiPoint(points).convex_hull

    feature = {
        "type": "Feature",
        "geometry": mapping(hull),
        "properties": {
            "region_id": region_id,
            "n_buses": len(points),
            "alpha_ratio": alpha_ratio,
            "source": "smart_ds Buscoords.dss (concave hull, per ADR-0024 amended)",
        },
    }
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [feature],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload) + "\n")

    return {
        "region_id": region_id,
        "n_buses": len(points),
        "alpha_ratio": alpha_ratio,
        "convex_hull_area": float(convex.area),
        "concave_hull_area": float(hull.area),
        "output_path": str(output_path),
    }


def iter_asset_registry_points(
    asset_registry_dir: Path,
) -> Iterator[tuple[float, float]]:
    """Yield (lon, lat) from every CSV in a Marshfield asset registry.

    Handles the four coord-column conventions used by the registry:
    ``lon/lat`` (buses, loads, sources), ``location_lon/location_lat``
    (transformers), and ``from_lon/from_lat`` + ``to_lon/to_lat`` (lines).
    CSVs lacking any coord column (e.g. ``feeders.csv``) yield nothing.
    """
    asset_registry_dir = Path(asset_registry_dir)
    for path in sorted(asset_registry_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for lon_key, lat_key in _ASSET_COORD_COLUMNS:
                    lon = row.get(lon_key)
                    lat = row.get(lat_key)
                    if lon in (None, "") or lat in (None, ""):
                        continue
                    try:
                        yield float(lon), float(lat)
                    except ValueError:
                        continue


def write_marshfield_power_extent(
    *,
    asset_registry_dir: Path,
    output_path: Path,
    alpha_ratio: float,
) -> dict:
    """Walk the asset registry and write a concave-hull power_extent.geojson.

    Returns a manifest dict for audit/indexing.
    """
    points = list(iter_asset_registry_points(asset_registry_dir))
    if not points:
        raise FileNotFoundError(
            f"no asset coordinates found under {asset_registry_dir}"
        )

    hull = concave_power_extent(points, alpha_ratio=alpha_ratio)
    convex = MultiPoint(points).convex_hull

    feature = {
        "type": "Feature",
        "geometry": mapping(hull),
        "properties": {
            "region_id": "marshfield",
            "n_assets": len(points),
            "alpha_ratio": alpha_ratio,
            "source": (
                "marshfield asset_registry (concave hull, per ADR-0024 amended)"
            ),
        },
    }
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [feature],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload) + "\n")

    return {
        "region_id": "marshfield",
        "n_assets": len(points),
        "alpha_ratio": alpha_ratio,
        "convex_hull_area": float(convex.area),
        "concave_hull_area": float(hull.area),
        "output_path": str(output_path),
    }


@dataclass(frozen=True)
class SfincsDomain:
    domain_id: str
    polygon: Polygon  # axis-aligned bounding box in EPSG:4326
    n_assets_inside: int
    aabb_km_x: float
    aabb_km_y: float


def _utm_epsg_for(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def cluster_to_sfincs_domains(
    points: Iterable[tuple[float, float]],
    *,
    region_id: str,
    alpha_split: float = 0.02,
    min_component_area_km2: float = 5.0,
    aabb_buffer_km: float = 1.0,
) -> list[SfincsDomain]:
    """Cluster ``points`` and emit one AABB per cluster, per [ADR-0037].

    1. Compute a tighter concave hull at ``alpha_split``. The tighter alpha
       lets disjoint clusters (Greensboro lobes, SFO Pacific-vs-Sacramento)
       fall apart into MultiPolygon components.
    2. Drop components below ``min_component_area_km2`` (small disconnected
       feeders that do not warrant a dedicated SFINCS run).
    3. For each surviving component compute the AABB in the local UTM CRS,
       buffer by ``aabb_buffer_km``, project back to EPSG:4326.
    """
    pts = list(points)
    if len(pts) < 3:
        raise ValueError(f"need at least 3 points to cluster; got {len(pts)}")

    hull = concave_power_extent(pts, alpha_ratio=alpha_split)
    components: list[Polygon]
    if hull.geom_type == "MultiPolygon":
        components = list(hull.geoms)
    elif hull.geom_type == "Polygon":
        components = [hull]
    else:
        raise TypeError(f"unexpected hull geometry {hull.geom_type}")

    # Project to UTM for accurate area + AABB calculation.
    centroid = MultiPoint(pts).centroid
    utm = _utm_epsg_for(centroid.x, centroid.y)
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform
    to_geo = Transformer.from_crs(utm, "EPSG:4326", always_xy=True).transform

    domains: list[SfincsDomain] = []
    for idx, component in enumerate(sorted(components, key=lambda g: -g.area)):
        comp_utm = shapely_transform(to_utm, component)
        area_km2 = comp_utm.area / 1e6
        if area_km2 < min_component_area_km2:
            continue
        xmin, ymin, xmax, ymax = comp_utm.bounds
        buf_m = aabb_buffer_km * 1000.0
        bbox_utm = box(xmin - buf_m, ymin - buf_m, xmax + buf_m, ymax + buf_m)
        bbox_geo = shapely_transform(to_geo, bbox_utm)
        n_inside = sum(
            1 for lon, lat in pts if component.covers(shapely.geometry.Point(lon, lat))
        )
        domains.append(
            SfincsDomain(
                domain_id=f"{region_id}:{idx}",
                polygon=bbox_geo,
                n_assets_inside=n_inside,
                aabb_km_x=(xmax - xmin + 2 * buf_m) / 1000.0,
                aabb_km_y=(ymax - ymin + 2 * buf_m) / 1000.0,
            )
        )
    return domains


def cluster_smart_ds_by_subregion(
    smart_ds_year_root: Path,
    *,
    region_id: str,
    min_n_buses: int = 1000,
    aabb_buffer_km: float = 1.0,
) -> list[SfincsDomain]:
    """Group SMART-DS buses by **subregion folder** and emit one AABB per
    subregion. This is the v0 clustering primitive — geometric auto-clustering
    via alpha-shape fails because SMART-DS rural feeders provide continuous
    connectivity between geographically distinct service areas.

    Subregion folders (P1U, P2U, ..., urban-suburban, rural, industrial, ...)
    are the committed structural grouping of the dataset and are stable across
    revisions. Any spatial merging of overlapping subregion AABBs is the
    caller's responsibility and lives in ``case.yaml``.
    """
    year_root = Path(smart_ds_year_root)
    subregions = sorted(d.name for d in year_root.iterdir() if d.is_dir())
    domains: list[SfincsDomain] = []
    # Project everything in a single UTM zone keyed on the first non-empty
    # subregion's centroid; consistent across all subregions in the region.
    seed_pts = None
    for sr in subregions:
        pts = list(iter_buscoords(year_root / sr))
        if pts:
            seed_pts = pts
            break
    if seed_pts is None:
        return domains
    seed_centroid = MultiPoint(seed_pts).centroid
    utm = _utm_epsg_for(seed_centroid.x, seed_centroid.y)
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform
    to_geo = Transformer.from_crs(utm, "EPSG:4326", always_xy=True).transform

    for sr in subregions:
        pts = list(iter_buscoords(year_root / sr))
        if len(pts) < min_n_buses:
            continue
        xs_utm, ys_utm = zip(*(to_utm(lon, lat) for lon, lat in pts))
        xmin, xmax = min(xs_utm), max(xs_utm)
        ymin, ymax = min(ys_utm), max(ys_utm)
        buf = aabb_buffer_km * 1000.0
        bbox_utm = box(xmin - buf, ymin - buf, xmax + buf, ymax + buf)
        bbox_geo = shapely_transform(to_geo, bbox_utm)
        domains.append(
            SfincsDomain(
                domain_id=f"{region_id}:{sr}",
                polygon=bbox_geo,
                n_assets_inside=len(pts),
                aabb_km_x=(xmax - xmin + 2 * buf) / 1000.0,
                aabb_km_y=(ymax - ymin + 2 * buf) / 1000.0,
            )
        )
    return domains


def cluster_marshfield_by_feeder(
    asset_registry_dir: Path,
    *,
    region_id: str = "marshfield",
    min_n_assets: int = 100,
    aabb_buffer_km: float = 1.0,
) -> list[SfincsDomain]:
    """Marshfield equivalent of subregion clustering: group asset-registry
    rows by the ``feeder_id`` column and emit one AABB per feeder."""
    asset_registry_dir = Path(asset_registry_dir)
    by_feeder: dict[str, list[tuple[float, float]]] = {}
    for path in sorted(asset_registry_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                feeder_id = row.get("feeder_id")
                if not feeder_id:
                    continue
                for lon_key, lat_key in _ASSET_COORD_COLUMNS:
                    lon = row.get(lon_key)
                    lat = row.get(lat_key)
                    if lon in (None, "") or lat in (None, ""):
                        continue
                    try:
                        by_feeder.setdefault(feeder_id, []).append((float(lon), float(lat)))
                    except ValueError:
                        continue
    if not by_feeder:
        return []
    # Single UTM zone keyed on the first feeder's centroid.
    seed_pts = next(iter(by_feeder.values()))
    seed_centroid = MultiPoint(seed_pts).centroid
    utm = _utm_epsg_for(seed_centroid.x, seed_centroid.y)
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform
    to_geo = Transformer.from_crs(utm, "EPSG:4326", always_xy=True).transform

    domains: list[SfincsDomain] = []
    for feeder_id, pts in sorted(by_feeder.items()):
        if len(pts) < min_n_assets:
            continue
        xs_utm, ys_utm = zip(*(to_utm(lon, lat) for lon, lat in pts))
        xmin, xmax = min(xs_utm), max(xs_utm)
        ymin, ymax = min(ys_utm), max(ys_utm)
        buf = aabb_buffer_km * 1000.0
        bbox_utm = box(xmin - buf, ymin - buf, xmax + buf, ymax + buf)
        bbox_geo = shapely_transform(to_geo, bbox_utm)
        domains.append(
            SfincsDomain(
                domain_id=f"{region_id}:{feeder_id}",
                polygon=bbox_geo,
                n_assets_inside=len(pts),
                aabb_km_x=(xmax - xmin + 2 * buf) / 1000.0,
                aabb_km_y=(ymax - ymin + 2 * buf) / 1000.0,
            )
        )
    return domains


def merge_overlapping_aabbs(
    domains: list[SfincsDomain],
    *,
    min_intersection_km2: float = 10.0,
    max_anchor_aabb_km: float | None = 80.0,
) -> list[SfincsDomain]:
    """Union-find merge of overlapping AABBs (per ADR-0037).

    Each connected component (transitively via pairwise polygon intersection)
    collapses into one merged domain whose AABB is the union of its members'
    AABBs in EPSG:4326. The merged ``domain_id`` is the lexicographically
    smallest member id, suffixed with ``" + N more"`` when the component has
    more than one member.

    Pairwise intersection is filtered by ``min_intersection_km2`` to avoid
    "kissing" merges caused by AABB buffer artifacts. The default 10 km²
    cleanly separates Greensboro's `urban-suburban` ∩ `rural` (~0.4 km²
    buffer-kiss) from `urban-suburban` ∩ `industrial` (~34 km² real overlap).

    ``max_anchor_aabb_km`` (default 80 km) keeps geographically dispersed
    AABBs from acting as merge bridges. SMART-DS rural subregions like SFO
    `P1R` have 170×170 km AABBs that span the whole study area and contain
    most urban subregions — without this guard they collapse the entire
    region into a single domain. Set to ``None`` to disable the guard.
    """
    if not domains:
        return []

    n = len(domains)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    def _is_anchor_eligible(d: SfincsDomain) -> bool:
        if max_anchor_aabb_km is None:
            return True
        return d.aabb_km_x <= max_anchor_aabb_km and d.aabb_km_y <= max_anchor_aabb_km

    for i in range(n):
        for j in range(i + 1, n):
            if not (_is_anchor_eligible(domains[i]) and _is_anchor_eligible(domains[j])):
                continue
            if not domains[i].polygon.intersects(domains[j].polygon):
                continue
            inter = domains[i].polygon.intersection(domains[j].polygon)
            # Convert deg² → km² using the joint centroid latitude.
            centroid_lat = (
                domains[i].polygon.centroid.y + domains[j].polygon.centroid.y
            ) / 2
            km2_per_deg2 = (111.0 ** 2) * math.cos(math.radians(centroid_lat))
            if inter.area * km2_per_deg2 >= min_intersection_km2:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: list[SfincsDomain] = []
    for members in groups.values():
        member_doms = [domains[i] for i in members]
        if len(member_doms) == 1:
            merged.append(member_doms[0])
            continue
        ids_sorted = sorted(d.domain_id for d in member_doms)
        xmin = min(d.polygon.bounds[0] for d in member_doms)
        ymin = min(d.polygon.bounds[1] for d in member_doms)
        xmax = max(d.polygon.bounds[2] for d in member_doms)
        ymax = max(d.polygon.bounds[3] for d in member_doms)
        bbox_geo = box(xmin, ymin, xmax, ymax)
        center_lat = (ymin + ymax) / 2
        km_x = (xmax - xmin) * 111.0 * math.cos(math.radians(center_lat))
        km_y = (ymax - ymin) * 111.0
        merged.append(
            SfincsDomain(
                domain_id=f"{ids_sorted[0]} + {len(member_doms) - 1} more",
                polygon=bbox_geo,
                n_assets_inside=sum(d.n_assets_inside for d in member_doms),
                aabb_km_x=km_x,
                aabb_km_y=km_y,
            )
        )
    # Stable order: by descending n_assets so the headline domain shows first.
    merged.sort(key=lambda d: -d.n_assets_inside)
    return merged


def merge_by_centroid_proximity(
    domains: list[SfincsDomain],
    *,
    eps_km: float = 20.0,
    min_assets_per_cluster: int = 50_000,
) -> list[SfincsDomain]:
    """Cluster sub-domains by **centroid distance in UTM metres**, then union
    each cluster's AABBs (per ADR-0037).

    This is the v0 default for SFO and the recommended method whenever a
    region has more than ~5 candidate subregions. It avoids the two failure
    modes of AABB-intersection merging:

    - Buffer-kiss false positives (Greensboro `rural` ↔ `urban-suburban`,
      0.4 km² overlap that crossed the intersection threshold).
    - Oversized-anchor false negatives (SFO `P1R` is 170×172 km and contained
      most other subregions, bridging everything into one component).

    ``eps_km`` is the maximum centroid-to-centroid distance for two
    subregions to be in the same SFINCS sub-domain. 20 km is the default —
    matches dense urban subregion spacing (SFO Bay), separates Greensboro's
    24-km-apart lobes, and keeps Austin's urban core as one cluster.

    ``min_assets_per_cluster`` drops orphan clusters below the threshold;
    these are usually rural P*R subregions with sparse buses that do not
    warrant a dedicated SFINCS run.
    """
    if not domains:
        return []

    # Anchor UTM zone on the region's geometric centroid for consistency.
    region_cx = sum(d.polygon.centroid.x for d in domains) / len(domains)
    region_cy = sum(d.polygon.centroid.y for d in domains) / len(domains)
    utm = _utm_epsg_for(region_cx, region_cy)
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform

    utm_centroids = [
        to_utm(d.polygon.centroid.x, d.polygon.centroid.y) for d in domains
    ]
    eps_m = eps_km * 1000.0

    n = len(domains)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            dx = utm_centroids[i][0] - utm_centroids[j][0]
            dy = utm_centroids[i][1] - utm_centroids[j][1]
            if math.hypot(dx, dy) < eps_m:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: list[SfincsDomain] = []
    for members in groups.values():
        member_doms = [domains[i] for i in members]
        total_assets = sum(d.n_assets_inside for d in member_doms)
        if total_assets < min_assets_per_cluster:
            continue
        if len(member_doms) == 1:
            merged.append(member_doms[0])
            continue
        ids_sorted = sorted(d.domain_id for d in member_doms)
        xmin = min(d.polygon.bounds[0] for d in member_doms)
        ymin = min(d.polygon.bounds[1] for d in member_doms)
        xmax = max(d.polygon.bounds[2] for d in member_doms)
        ymax = max(d.polygon.bounds[3] for d in member_doms)
        bbox_geo = box(xmin, ymin, xmax, ymax)
        center_lat = (ymin + ymax) / 2
        km_x = (xmax - xmin) * 111.0 * math.cos(math.radians(center_lat))
        km_y = (ymax - ymin) * 111.0
        merged.append(
            SfincsDomain(
                domain_id=f"{ids_sorted[0]} + {len(member_doms) - 1} more",
                polygon=bbox_geo,
                n_assets_inside=total_assets,
                aabb_km_x=km_x,
                aabb_km_y=km_y,
            )
        )
    merged.sort(key=lambda d: -d.n_assets_inside)
    return merged


def cluster_buses_kmeans(
    points: Iterable[tuple[float, float]],
    *,
    region_id: str,
    k: int,
    aabb_buffer_km: float = 1.0,
    seed: int = 0,
) -> list[SfincsDomain]:
    """Bus-level k-means clustering — the v0 default for sub-domain derivation.

    Cluster every asset coord (subsampled or full) directly into ``k`` spatial
    groups using k-means in the region's UTM CRS, then compute one AABB per
    cluster from **only that cluster's assets** (buffered by ``aabb_buffer_km``).

    Why this method (per ADR-0037, after iteration):

    - Subregion-based clustering + AABB-intersection merge: false positives
      from 1 km buffer kisses; false negatives from oversized rural anchors.
    - Subregion-based + centroid-distance merge: subregions are 20-50 km wide;
      adjacent clusters' subregion-AABBs still overlap heavily.
    - Bus-level k-means: each bus belongs to exactly one cluster, and the
      cluster's AABB is tight to its asset subset. Adjacent cluster AABBs
      can still overlap at the boundary where assets interleave, but the
      overlap is minimized (only the edge-of-cluster buses bleed across).

    ``k`` is per-region and lives in ``case.yaml``. Recommended defaults
    (matching the visual lobes in notebooks/regions/bbox.ipynb):

    | region | k | reason |
    |---|---|---|
    | marshfield | 1 | single municipality |
    | austin | 1 | one urban core, all subregions overlap centroids |
    | greensboro | 2 | west lobe (urban+industrial) vs. east lobe (rural) |
    | sfo | 4 | SF Bay, North Bay (Marin/Sonoma), East Bay/Sacramento, South Bay/Salinas |
    """
    pts = list(points)
    if len(pts) < k:
        raise ValueError(
            f"k={k} requested but only {len(pts)} points provided"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")

    centroid = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    utm = _utm_epsg_for(*centroid)
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform
    to_geo = Transformer.from_crs(utm, "EPSG:4326", always_xy=True).transform

    pts_utm = np.array([to_utm(lon, lat) for lon, lat in pts])
    if k == 1:
        labels = np.zeros(len(pts), dtype=int)
    else:
        _, labels = kmeans2(pts_utm, k, seed=seed, minit="++")

    domains: list[SfincsDomain] = []
    buf_m = aabb_buffer_km * 1000.0
    for cluster_idx in range(k):
        mask = labels == cluster_idx
        if not mask.any():
            continue
        cluster_utm = pts_utm[mask]
        xmin, xmax = float(cluster_utm[:, 0].min()), float(cluster_utm[:, 0].max())
        ymin, ymax = float(cluster_utm[:, 1].min()), float(cluster_utm[:, 1].max())
        bbox_utm = box(xmin - buf_m, ymin - buf_m, xmax + buf_m, ymax + buf_m)
        bbox_geo = shapely_transform(to_geo, bbox_utm)
        domains.append(
            SfincsDomain(
                domain_id=f"{region_id}:k{cluster_idx}",
                polygon=bbox_geo,
                n_assets_inside=int(mask.sum()),
                aabb_km_x=(xmax - xmin + 2 * buf_m) / 1000.0,
                aabb_km_y=(ymax - ymin + 2 * buf_m) / 1000.0,
            )
        )
    domains.sort(key=lambda d: -d.n_assets_inside)
    return domains


def write_sfincs_domains(
    *,
    region_id: str,
    points: Iterable[tuple[float, float]],
    output_path: Path,
    alpha_split: float = 0.02,
    min_component_area_km2: float = 5.0,
    aabb_buffer_km: float = 1.0,
) -> list[SfincsDomain]:
    domains = cluster_to_sfincs_domains(
        points,
        region_id=region_id,
        alpha_split=alpha_split,
        min_component_area_km2=min_component_area_km2,
        aabb_buffer_km=aabb_buffer_km,
    )
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(d.polygon),
                "properties": {
                    "domain_id": d.domain_id,
                    "n_assets_inside": d.n_assets_inside,
                    "aabb_km_x": round(d.aabb_km_x, 1),
                    "aabb_km_y": round(d.aabb_km_y, 1),
                    "source": (
                        f"cluster-derived AABB at alpha_split={alpha_split} "
                        "per ADR-0037"
                    ),
                },
            }
            for d in domains
        ],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload) + "\n")
    return domains
