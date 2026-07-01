"""Wflow-SFINCS Domain Set manifest writer."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

from paths import location_root_from_paths, relative_to_or_absolute, resolve_location_path
from coupling.handoff_sources import read_stream_boundary_handoff_location_artifacts
import wflow_runs.types as wflow_types


_GENERATED_NOTICE = (
    "# GENERATED FILE — do not edit. Overwritten when {source} runs.\n"
    "# Source of truth is the location config and the code that produces this file.\n"
)


def write_wflow_domain_set_manifest(plan: wflow_types.WflowDomainSetPlan, config, paths) -> Path:
    """Write the reviewed Wflow-SFINCS Domain Set manifest."""
    if plan.status != "ready":
        raise ValueError(f"Wflow Domain Set plan is not ready: {plan.status}")
    location_root = location_root_from_paths(paths)
    wflow = config.get("wflow", {})
    submodels = _manifest_submodels_from_active_handoff_sources(plan, config, location_root)
    manifest_path = resolve_location_path(
        location_root,
        wflow.get("domain_set_manifest", "data/wflow/domain_set.yaml"),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sfincs_domains_by_handoff = _sfincs_domain_ids_by_handoff(config, location_root)
    manifest = {
        "event_catalog_scope": wflow.get("domain_set", {}).get(
            "event_catalog_scope",
            "shared_across_domain_set",
        ),
        "reviewed_network": relative_to_or_absolute(plan.reviewed_network, location_root),
        "subbasin_fabric": wflow.get("domain_set", {}).get(
            "subbasin_fabric",
            "data/wflow/domain_set_subbasins.gpkg",
        ),
        "handoff": {
            "source_variable": wflow.get("handoff", {}).get("source_variable", "river_q"),
            "source_standard_name": wflow.get("handoff", {}).get(
                "source_standard_name",
                "river_water__volume_flow_rate",
            ),
            "target": wflow.get("handoff", {}).get("target", "sfincs_discharge_forcing"),
            "sfincs_boundary_type": "discharge",
        },
        "submodels": [
            {
                "wflow_submodel_id": submodel["wflow_submodel_id"],
                "hydromt_region": submodel["region"],
                "handoff_outlet_region": submodel.get("outlet_region", submodel["region"]),
                "region_kind": submodel.get("region_kind"),
                "subbasin_geometry": relative_to_or_absolute(Path(submodel["subbasin_geometry"]), location_root)
                if submodel.get("subbasin_geometry")
                else None,
                "sfincs_domain_ids": _manifest_sfincs_domain_ids(submodel, sfincs_domains_by_handoff),
                "sfincs_handoff_ids": list(submodel["sfincs_handoff_ids"]),
                "sfincs_boundary_ids": list(submodel["sfincs_handoff_ids"]),
                "gauge_site_nos": list(submodel["gauge_site_nos"]),
                "frequency_basis": list(submodel["frequency_basis"]),
                "role_counts": dict(submodel["role_counts"]),
            }
            for submodel in submodels
        ],
    }
    manifest_path.write_text(
        _GENERATED_NOTICE.format(source="the Wflow domain-set build")
        + yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path


def _manifest_submodels_from_active_handoff_sources(
    plan: wflow_types.WflowDomainSetPlan,
    config: dict,
    location_root: Path,
) -> tuple[dict, ...]:
    """Use generated SFINCS source artifacts as the final stream-boundary handoff IDs."""
    if config.get("wflow", {}).get("domain_set", {}).get("ignore_sfincs_handoff_artifacts"):
        return plan.submodels
    locations = read_stream_boundary_handoff_location_artifacts(
        config,
        location_root,
        location_path=resolve_location_path,
    )
    if locations is None or locations.empty or "wflow_submodel_id" not in locations:
        return plan.submodels

    by_submodel: dict[str, gpd.GeoDataFrame] = {}
    for submodel_id, group in locations.groupby(locations["wflow_submodel_id"].astype(str), sort=True):
        by_submodel[str(submodel_id)] = group.copy()
    if not by_submodel:
        return plan.submodels

    synced = []
    for submodel in plan.submodels:
        submodel_id = str(submodel.get("wflow_submodel_id", ""))
        group = by_submodel.get(submodel_id)
        if group is None or group.empty:
            synced.append(submodel)
            continue
        group = group.sort_values("sfincs_handoff_id").reset_index(drop=True)
        ids = [str(value) for value in group["sfincs_handoff_id"]]
        updated = dict(submodel)
        updated["sfincs_handoff_ids"] = ids
        updated["sfincs_boundary_ids"] = ids
        if "sfincs_domain_id" in group:
            updated["sfincs_domain_ids"] = _sorted_values(group["sfincs_domain_id"].astype(str))
        uparea_col = "uparea_km2" if "uparea_km2" in group else "uparea" if "uparea" in group else None
        group_wgs = group.to_crs("EPSG:4326") if group.crs is not None else group
        updated["handoff_points"] = [
            {
                "sfincs_handoff_id": str(row["sfincs_handoff_id"]),
                "sfincs_domain_id": str(row.get("sfincs_domain_id", "")),
                "lon": float(row.geometry.x),
                "lat": float(row.geometry.y),
                "uparea_km2": float(row[uparea_col]) if uparea_col and pd.notna(row[uparea_col]) else float("nan"),
            }
            for _, row in group_wgs.iterrows()
        ]
        synced.append(updated)
    return tuple(synced)


def _manifest_sfincs_domain_ids(submodel: dict, sfincs_domains_by_handoff: dict[str, tuple[str, ...]]) -> list[str]:
    domain_ids = set()
    for handoff_id in submodel.get("sfincs_handoff_ids", ()):
        domain_ids.update(sfincs_domains_by_handoff.get(str(handoff_id), ()))
    if domain_ids:
        return sorted(domain_ids)
    return list(submodel["sfincs_domain_ids"])


def _sfincs_domain_ids_by_handoff(config, location_root: Path) -> dict[str, tuple[str, ...]]:
    manifest_value = config.get("sfincs_domain_set", {}).get(
        "domain_manifest",
        "data/sfincs/domains/domain_set.yaml",
    )
    manifest_path = resolve_location_path(location_root, manifest_value)
    if not manifest_path.exists():
        return {}
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    handoff_domains: dict[str, set[str]] = {}
    for domain in manifest.get("domains", []):
        domain_id = domain.get("sfincs_domain_id")
        if not domain_id:
            continue
        for handoff_id in domain.get("handoff_source_ids", []):
            handoff_domains.setdefault(str(handoff_id), set()).add(str(domain_id))
    return {
        handoff_id: tuple(sorted(domain_ids))
        for handoff_id, domain_ids in handoff_domains.items()
    }


def _sorted_values(values) -> tuple:
    return tuple(sorted({str(value) for value in values if value not in {None, ""}}))
