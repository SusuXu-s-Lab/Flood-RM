from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paths import location_or_repo_path_from_paths, resolve_location_path


@dataclass(frozen=True)
class StructureLayer:
    name: str
    kind: str
    component: str
    path: Path
    feature_count: int
    merge: bool
    dz: float | None = None
    stype: str | None = None
    drop_reason: str = ""


@dataclass(frozen=True)
class StructurePlan:
    study_location: str
    applied_layers: tuple[StructureLayer, ...]
    dropped_layers: tuple[StructureLayer, ...]

    def summary_rows(self):
        rows = []
        for layer in self.applied_layers:
            rows.append(
                {
                    "layer": layer.name,
                    "decision": "applied",
                    "kind": layer.kind,
                    "feature_count": layer.feature_count,
                    "path": layer.path.as_posix(),
                    "reason": "",
                }
            )
        for layer in self.dropped_layers:
            rows.append(
                {
                    "layer": layer.name,
                    "decision": "dropped",
                    "kind": layer.kind,
                    "feature_count": layer.feature_count,
                    "path": layer.path.as_posix(),
                    "reason": layer.drop_reason,
                }
            )
        return rows


def prepare_structure_layers(config: dict[str, Any], paths: dict[str, Any]) -> StructurePlan:
    structures = config.get("sfincs_structures") or {}
    if not structures.get("enabled", False):
        return StructurePlan(_study_location(config, paths), (), ())

    source_root = _resolve_path(paths, structures.get("source_root", "data/static/structures/sources"))
    output_root = _resolve_location_path(paths, structures.get("output_root", "data/static/structures/screening"))

    applied: list[StructureLayer] = []
    dropped: list[StructureLayer] = []
    for entry in structures.get("layers") or []:
        component = str(entry["component"])
        source = _resolve_source(source_root, entry["source"])
        enabled = bool(entry.get("enabled", True))
        if not enabled:
            continue
        target = output_root / Path(entry["source"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

        layer = StructureLayer(
            name=str(entry["name"]),
            kind=str(entry.get("kind") or _kind_from_component(component)),
            component=component,
            path=target,
            feature_count=_feature_count(target),
            merge=bool(entry.get("merge", True)),
            dz=entry.get("dz"),
            stype=entry.get("stype"),
            drop_reason=str(entry.get("drop_reason", "")),
        )
        applied.append(layer)

    return StructurePlan(_study_location(config, paths), tuple(applied), tuple(dropped))


def apply_sfincs_structures(sf, plan: StructurePlan) -> StructurePlan:
    for layer in plan.applied_layers:
        if layer.component == "weirs":
            sf.weirs.create(locations=str(layer.path), dz=layer.dz, merge=layer.merge)
        elif layer.component == "thin_dams":
            sf.thin_dams.create(locations=str(layer.path), merge=layer.merge)
        elif layer.component == "drainage_structures":
            kwargs = {"locations": str(layer.path), "merge": layer.merge}
            if layer.stype:
                kwargs["stype"] = layer.stype
            sf.drainage_structures.create(**kwargs)
        else:
            raise ValueError(f"Unsupported SFINCS structure component: {layer.component}")
    return plan


def plot_structure_layers(
    plan: StructurePlan,
    *,
    output_path: str | Path | None = None,
    ax=None,
    map_context: bool = False,
    single_color: str | None = None,
):
    import geopandas as gpd
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7)) if ax is None and map_context else (
        plt.subplots(figsize=(9, 8)) if ax is None else (ax.figure, ax)
    )
    styles = {
        "weir": {"color": "#d95f02", "linewidth": 2.0, "label": "weirs"},
        "thin_dam": {"color": "#1b9e77", "linewidth": 2.0, "label": "thin dams"},
        "drainage_structure": {"color": "#7570b3", "linewidth": 1.5, "label": "drainage structures"},
    }

    plotted_labels = set()
    bounds = []
    for layer in plan.applied_layers:
        gdf = gpd.read_file(layer.path)
        gdf = _map_gdf(gdf) if map_context else gdf
        style = styles.get(layer.kind, {"color": "#333333", "linewidth": 1.5, "label": layer.kind})
        color = single_color or style["color"]
        label = None if map_context else style["label"] if style["label"] not in plotted_labels else None
        gdf.plot(
            ax=ax,
            color=color,
            linewidth=2.2 if map_context else style["linewidth"],
            label=label,
            zorder=5,
        )
        bounds.append(gdf.total_bounds)
        plotted_labels.add(style["label"])

    for layer in plan.dropped_layers:
        gdf = gpd.read_file(layer.path)
        gdf = _map_gdf(gdf) if map_context else gdf
        label = None if map_context else "dropped" if "dropped" not in plotted_labels else None
        gdf.plot(ax=ax, color="#999999", linewidth=1.0, linestyle="--", alpha=0.65, label=label, zorder=4)
        bounds.append(gdf.total_bounds)
        plotted_labels.add("dropped")

    if map_context:
        _set_padded_extent(ax, bounds, pad_fraction=0.22)
        _add_light_basemap(ax)
        ax.set_axis_off()
    else:
        ax.set_title(f"{plan.study_location.title()} SFINCS structure layers")
        ax.set_xlabel("Longitude / Easting")
        ax.set_ylabel("Latitude / Northing")
        ax.set_aspect("equal")
        if plotted_labels:
            ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return fig, ax


def _map_gdf(gdf):
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(epsg=3857)


def _set_padded_extent(ax, bounds: list[Any], *, pad_fraction: float) -> None:
    if not bounds:
        return
    minx = min(float(bound[0]) for bound in bounds)
    miny = min(float(bound[1]) for bound in bounds)
    maxx = max(float(bound[2]) for bound in bounds)
    maxy = max(float(bound[3]) for bound in bounds)
    dx = max(maxx - minx, 1.0)
    dy = max(maxy - miny, 1.0)
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    side = max(dx, dy) * (1 + 2 * pad_fraction)
    ax.set_xlim(center_x - side / 2, center_x + side / 2)
    ax.set_ylim(center_y - side / 2, center_y + side / 2)


def _add_light_basemap(ax) -> None:
    ax.set_facecolor("#e6e7eb")
    try:
        import contextily as ctx

        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
            attribution=False,
            zoom="auto",
        )
    except Exception:
        # Tests and air-gapped runs should still produce a useful structure QA plot.
        ax.grid(color="white", linewidth=0.8, alpha=0.75)


def derive_massgis_sfincs_structure_layers(
    source: str | Path,
    output_root: str | Path,
    *,
    weirs_name: str = "weirs_marshfield_massgis_public_2015.geojson",
    thin_dams_name: str = "thin_dams_marshfield_massgis_public_2015.geojson",
) -> dict[str, Path]:
    source = Path(source)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    data = json.loads(source.read_text(encoding="utf-8"))
    weirs = []
    thin_dams = []
    omitted = []

    for feature in data.get("features") or []:
        properties = feature.get("properties") or {}
        primary_type = _normalize_massgis_text(properties.get("PrimaryTyp"))
        if primary_type in {"bulkhead/ seawall", "bulkhead/seawall", "revetment"}:
            z_ft = _parse_float(properties.get("PositionZ"))
            if z_ft is None:
                omitted.append(_massgis_summary(properties, "omit_weir_without_positionz"))
                continue
            weirs.append(_massgis_weir_feature(feature, z_ft))
        elif primary_type in {"groin/ jetty", "groin/jetty", "jetty/groin"}:
            thin_dams.append(_massgis_thin_dam_feature(feature))
        else:
            omitted.append(_massgis_summary(properties, "omit_non_sfincs_structure_type"))

    weirs_path = output_root / weirs_name
    thin_dams_path = output_root / thin_dams_name
    summary_path = output_root / "massgis_public_2015_sfincs_derivation_summary.json"
    weirs_path.write_text(json.dumps(_feature_collection(weirs), indent=2), encoding="utf-8")
    thin_dams_path.write_text(json.dumps(_feature_collection(thin_dams), indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "source": source.as_posix(),
                "weir_features": len(weirs),
                "thin_dam_features": len(thin_dams),
                "omitted_features": len(omitted),
                "omitted": omitted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"weirs": weirs_path, "thin_dams": thin_dams_path, "summary": summary_path}


def _resolve_source(source_root: Path, source: str) -> Path:
    path = Path(source)
    if path.is_absolute():
        return path
    return source_root / path


def _kind_from_component(component: str) -> str:
    if component == "weirs":
        return "weir"
    if component == "thin_dams":
        return "thin_dam"
    if component == "drainage_structures":
        return "drainage_structure"
    return component


def _resolve_path(paths: dict[str, Any], value: str | Path) -> Path:
    return location_or_repo_path_from_paths(paths, value)


def _resolve_location_path(paths: dict[str, Any], value: str | Path) -> Path:
    return resolve_location_path(paths["location_root"], value)


def _study_location(config: dict[str, Any], paths: dict[str, Any]) -> str:
    return str(paths.get("location_name") or config.get("project", {}).get("name", ""))


def _feature_count(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data.get("features") or [])


def _feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def _massgis_weir_feature(feature: dict[str, Any], z_ft: float) -> dict[str, Any]:
    properties = feature.get("properties") or {}
    out = _massgis_base_feature(feature, "weir")
    out["properties"].update(
        {
            "sfincs_kind": "weir",
            "hydromt_stype": "weir",
            "z": round(z_ft * 0.3048, 3),
            "z_units": "m",
            "z_ft_navd88": z_ft,
            "z_m_navd88": round(z_ft * 0.3048, 3),
            "par1": 0.6,
            "primary_visible_height_ft_band": _clean_massgis_text(properties.get("PrimaryHei")),
            "elevation_source": "MassGIS PositionZ lidar elevation field; verify against survey before design use.",
            "design_grade_elevation_source": "WHG/USACE/Town plan sheets or as-built survey.",
        }
    )
    return out


def _massgis_thin_dam_feature(feature: dict[str, Any]) -> dict[str, Any]:
    out = _massgis_base_feature(feature, "thin_dam")
    out["properties"].update(
        {
            "sfincs_kind": "thin_dam",
            "hydromt_stype": "thd",
            "use_note": "Use as a thin dam only where model resolution/topobathy allows cross-structure leakage.",
        }
    )
    return out


def _massgis_base_feature(feature: dict[str, Any], kind: str) -> dict[str, Any]:
    properties = feature.get("properties") or {}
    return {
        "type": "Feature",
        "properties": {
            "id": _clean_massgis_text(properties.get("STR_ID")),
            "name": _clean_massgis_text(properties.get("Location")),
            "source": "MassGIS/CZM Shoreline Stabilization Structures, Public Structures 2015 Update",
            "source_structure_id": _clean_massgis_text(properties.get("STR_ID")),
            "source_primary_type": _clean_massgis_text(properties.get("PrimaryTyp")),
            "source_secondary_type": _clean_massgis_text(properties.get("SecondaryT")),
            "source_primary_material": _clean_massgis_text(properties.get("PrimaryMat")),
            "source_primary_condition": _clean_massgis_text(properties.get("PrimaryCon")),
            "source_fema_zone": _clean_massgis_text(properties.get("FEMAZone")),
            "source_fema_elevation": _clean_massgis_text(properties.get("FEMAElev")),
            "source_waterway": _clean_massgis_text(properties.get("Waterway")),
            "source_length_ft": _parse_float(properties.get("SHAPE_Leng")),
            "vertical_datum": "NAVD88",
            "status": "screening_from_public_inventory_not_surveyed",
            "design_grade_geometry_source": "WHG/USACE/Town plan sheets or final CAD/as-built drawings.",
            "kind": kind,
        },
        "geometry": feature.get("geometry"),
    }


def _massgis_summary(properties: dict[str, Any], reason: str) -> dict[str, str]:
    return {
        "id": _clean_massgis_text(properties.get("STR_ID")),
        "location": _clean_massgis_text(properties.get("Location")),
        "primary_type": _clean_massgis_text(properties.get("PrimaryTyp")),
        "reason": reason,
    }


def _normalize_massgis_text(value: Any) -> str:
    return _clean_massgis_text(value).lower()


def _clean_massgis_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_float(value: Any) -> float | None:
    text = _clean_massgis_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
