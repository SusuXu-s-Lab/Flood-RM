"""Baseline source-area and native SHIFT/GDM/DiTTo adapters.

The only local responsibilities here are case configuration, reviewed source
anchors, and artifact locations.  OSM data pull, road-network acquisition,
parcel conversion, graph construction, phase/voltage mapping, equipment
mapping, and GDM system assembly are delegated to SHIFT/GDM/DiTTo native APIs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .native import (
    NativeDependencyError,
    load_shift_test_catalog,
    read_gdm_json,
    require_module,
    shift_distance,
    shift_geo_location,
    write_gdm_json,
    write_opendss,
)


@dataclass(frozen=True)
class GridSourceArea:
    place_name: str
    geometry: Any
    patch_names: tuple[str, ...]


@dataclass(frozen=True)
class SourceAnchorReviewRequired(RuntimeError):
    candidate_path: Path
    reviewed_path: Path

    def __str__(self) -> str:
        return (
            "review grid source anchors before building the Baseline Network: "
            f"copy reviewed candidates from {self.candidate_path} to {self.reviewed_path}"
        )


def source_area(config: dict[str, Any], *, geocode_place: Callable[[str], Any] | None = None) -> GridSourceArea:
    """Build a reviewed source-area polygon from OSMnx place geometry.

    SHIFT owns parcel and road pulls, but the case still needs a visible human
    review geometry for anchors and SFINCS domain choices.
    """

    from shapely.geometry import box
    from shapely.ops import unary_union

    if geocode_place is None:
        import osmnx as ox

        geocode_place = ox.geocode_to_gdf

    grid = config.get("grid") or {}
    spec = grid.get("source_area") or {}
    if spec.get("source") != "osmnx_place":
        raise ValueError(f"unsupported grid.source_area.source: {spec.get('source')!r}")
    place_name = _resolve_config_reference(config, spec.get("place_name", "project.place_name"))
    boundary = geocode_place(place_name).geometry.union_all()
    patches = []
    names: list[str] = []
    for patch in spec.get("extra_areas") or []:
        patches.append(box(*[float(value) for value in patch["bbox"]]))
        names.append(str(patch.get("name", "unnamed_patch")))
    return GridSourceArea(place_name=place_name, geometry=unary_union([boundary, *patches]), patch_names=tuple(names))


def source_anchors(
    config: dict[str, Any],
    *,
    location_root: str | Path,
    source_area_geometry: Any,
    fetch_candidates: Callable[[Any], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Return reviewed source anchors or write OSM candidates and stop."""

    spec = (config.get("grid") or {}).get("source_anchors") or {}
    reviewed_path = _location_path(Path(location_root), spec.get("reviewed_override"))
    candidate_path = _location_path(Path(location_root), spec.get("candidate_output"))
    if reviewed_path.exists():
        return _read_anchor_geojson(reviewed_path)

    fetch = fetch_candidates or _fetch_osm_power_substations
    candidates = _snap_anchors_to_source_area(fetch(source_area_geometry), source_area_geometry)
    _write_anchor_geojson(candidate_path, candidates)
    if spec.get("accept_unreviewed_source_anchors"):
        return candidates
    raise SourceAnchorReviewRequired(candidate_path=candidate_path, reviewed_path=reviewed_path)


def fetch_parcels_native(location: str | Any, *, max_distance_m: float = 500.0) -> list[Any] | None:
    """Fetch parcels through ``shift.parcels_from_location``.

    ``location`` may be a place string, ``shift.GeoLocation``, or a Shapely
    polygon.  SHIFT handles the OSMnx pull and conversion to ParcelModel.
    """

    shift = require_module("shift", package="nrel-shift", purpose="SHIFT parcel fetching")
    return shift.parcels_from_location(location, shift_distance(max_distance_m, "meter"))


def fetch_road_network_native(location: str | Any, *, max_distance_m: float = 500.0) -> Any:
    """Fetch the road network through ``shift.get_road_network``."""

    shift = require_module("shift", package="nrel-shift", purpose="SHIFT road-network fetching")
    return shift.get_road_network(location, shift_distance(max_distance_m, "meter"))


def cluster_parcels_native(parcels: Sequence[Any], *, customers_per_transformer: int = 2) -> Any:
    """Cluster parcel points through SHIFT's native K-means helper."""

    shift = require_module("shift", package="nrel-shift", purpose="SHIFT parcel clustering")
    points = [parcel.geometry[0] if isinstance(parcel.geometry, list) else parcel.geometry for parcel in parcels]
    cluster_count = max(len(points) // max(customers_per_transformer, 1), 1)
    return shift.get_kmeans_clusters(cluster_count, points)


def build_distribution_graph_native(groups: Any, *, source_location: Any) -> Any:
    """Build a SHIFT PRSG distribution graph."""

    shift = require_module("shift", package="nrel-shift", purpose="SHIFT graph construction")
    return shift.PRSG(groups=groups, source_location=source_location).get_distribution_graph()


def load_equipment_catalog(path: str | Path | None = None, *, shift_test_catalog: str = "p1rhs7_1247.json") -> Any:
    """Load a GDM equipment catalog with native GDM JSON deserialization."""

    if path is None:
        return load_shift_test_catalog(shift_test_catalog)
    return read_gdm_json(path)


def build_shift_distribution_system(
    *,
    name: str,
    parcels: Sequence[Any],
    source_longitude: float,
    source_latitude: float,
    equipment_catalog: Any | None = None,
    customers_per_transformer: int = 2,
    primary_voltage_kv: float = 7.2,
    secondary_voltage_v: float = 120.0,
    phase_method: str = "agglomerative",
) -> Any:
    """Build a GDM ``DistributionSystem`` using the documented SHIFT workflow."""

    shift = require_module("shift", package="nrel-shift", purpose="SHIFT baseline build")
    try:
        from gdm import DistributionBranchBase, DistributionTransformer, MatrixImpedanceBranch  # type: ignore
    except ImportError:
        from gdm.distribution.components import DistributionBranchBase, DistributionTransformer, MatrixImpedanceBranch  # type: ignore
    from gdm.quantities import ApparentPower, Voltage  # type: ignore

    groups = cluster_parcels_native(parcels, customers_per_transformer=customers_per_transformer)
    graph = build_distribution_graph_native(groups, source_location=shift_geo_location(source_longitude, source_latitude))
    graph = _with_matrix_impedance_branches(graph, DistributionBranchBase, MatrixImpedanceBranch)

    transformer_phase_models = [
        shift.TransformerPhaseMapperModel(
            tr_name=edge.name,
            tr_type=shift.TransformerTypes.SPLIT_PHASE,
            tr_capacity=ApparentPower(25, "kilovoltampere"),
            location=graph.get_node(from_node).location,
        )
        for from_node, _, edge in graph.get_edges()
        if edge.edge_type is DistributionTransformer
    ]
    phase_mapper = shift.BalancedPhaseMapper(graph, mapper=transformer_phase_models, method=phase_method)
    voltage_mapper = shift.TransformerVoltageMapper(
        graph,
        xfmr_voltage=[
            shift.TransformerVoltageModel(
                name=edge.name,
                voltages=[Voltage(primary_voltage_kv, "kilovolt"), Voltage(secondary_voltage_v, "volt")],
            )
            for _, _, edge in graph.get_edges()
            if edge.edge_type is DistributionTransformer
        ],
    )
    catalog = equipment_catalog or load_equipment_catalog(None)
    equipment_mapper = shift.EdgeEquipmentMapper(graph, catalog, voltage_mapper, phase_mapper)
    return shift.DistributionSystemBuilder(
        name=name,
        dist_graph=graph,
        phase_mapper=phase_mapper,
        voltage_mapper=voltage_mapper,
        equipment_mapper=equipment_mapper,
    ).get_system()


def export_baseline_system(
    system: Any,
    *,
    gdm_json: str | Path | None = None,
    opendss_dir: str | Path | None = None,
) -> dict[str, str]:
    """Export a native GDM system through GDM JSON and/or DiTTo OpenDSS."""

    outputs: dict[str, str] = {}
    if gdm_json is not None:
        outputs["gdm_json"] = str(write_gdm_json(system, gdm_json))
    if opendss_dir is not None:
        outputs["opendss_dir"] = str(write_opendss(system, opendss_dir))
    return outputs


def _with_matrix_impedance_branches(graph: Any, branch_base_type: type[Any], matrix_branch_type: type[Any]) -> Any:
    """Apply the branch-type conversion shown in the SHIFT complete example."""

    shift = require_module("shift", package="nrel-shift", purpose="SHIFT branch-type update")
    new_graph = shift.DistributionGraph()
    for node in graph.get_nodes():
        new_graph.add_node(node)
    for from_node, to_node, edge in graph.get_edges():
        if edge.edge_type == branch_base_type:
            edge.edge_type = matrix_branch_type
        new_graph.add_edge(from_node, to_node, edge_data=edge)
    return new_graph


def _resolve_config_reference(config: dict[str, Any], value: str) -> str:
    if value == "project.place_name":
        place_name = str((config.get("project") or {}).get("place_name", "")).strip()
        if not place_name:
            raise ValueError("project.place_name is required for grid source ingestion")
        return place_name
    return str(value)


def _location_path(location_root: Path, value: str | None) -> Path:
    if not value:
        raise ValueError("source anchor path is required")
    path = Path(value)
    return path if path.is_absolute() else location_root / path


def _snap_anchors_to_source_area(anchors: list[dict[str, Any]], source_area_geometry: Any) -> list[dict[str, Any]]:
    from shapely.geometry import Point
    from shapely.ops import nearest_points

    snapped = []
    for index, row in enumerate(anchors, start=1):
        point = Point(float(row["lon"]), float(row["lat"]))
        if not (source_area_geometry.contains(point) or source_area_geometry.touches(point)):
            point = nearest_points(source_area_geometry, point)[0]
        snapped.append({**row, "name": str(row.get("name") or f"source_anchor_{index}"), "substation_id": str(row.get("substation_id") or f"source_anchor_{index}"), "lon": float(point.x), "lat": float(point.y)})
    return snapped


def _write_anchor_geojson(path: Path, anchors: list[dict[str, Any]]) -> None:
    from shapely.geometry import Point, mapping

    path.parent.mkdir(parents=True, exist_ok=True)
    features = [
        {"type": "Feature", "geometry": mapping(Point(float(row["lon"]), float(row["lat"]))), "properties": {k: v for k, v in row.items() if k not in {"lon", "lat"}}}
        for row in anchors
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n", encoding="utf-8")


def _read_anchor_geojson(path: Path) -> list[dict[str, Any]]:
    from shapely.geometry import shape

    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors = []
    for index, feature in enumerate(payload.get("features", []), start=1):
        point = shape(feature["geometry"])
        props = dict(feature.get("properties") or {})
        anchors.append({**props, "name": str(props.get("name") or f"source_anchor_{index}"), "substation_id": str(props.get("substation_id") or f"source_anchor_{index}"), "lon": float(point.x), "lat": float(point.y)})
    return anchors


def _fetch_osm_power_substations(source_area_geometry: Any) -> list[dict[str, Any]]:
    import osmnx as ox

    anchors = []
    frame = ox.features_from_polygon(source_area_geometry, {"power": "substation"})
    for index, row in enumerate(frame.itertuples(), start=1):
        geom = row.geometry
        point = geom if geom.geom_type == "Point" else geom.representative_point()
        osm_id = getattr(row, "osmid", None) or getattr(row, "osm_id", None) or index
        anchors.append({"name": str(getattr(row, "name", "") or f"OSM substation {osm_id}"), "substation_id": f"osm:{osm_id}", "lon": float(point.x), "lat": float(point.y), "source": "osm_power_substations"})
    return anchors
