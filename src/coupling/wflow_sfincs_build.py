from __future__ import annotations

from pathlib import Path

import pandas as pd

from paths import resolve_location_path
from coupling.domain_manifest import write_wflow_domain_set_manifest
from coupling.domain_set import write_wflow_crossing_gauge_locations
from coupling.wflow_domain_set import plan_wflow_domain_set
from wflow_runs.fabric import write_wflow_subbasin_fabric_from_nhdplus
from wflow_runs.hydromt_build import build_wflow_build_plan


def exists_table(location_root: Path, named_paths: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "artifact": name,
                "path": str(resolve_location_path(location_root, relative_path)),
                "exists": resolve_location_path(location_root, relative_path).exists(),
            }
            for name, relative_path in named_paths.items()
        ]
    )


def domain_summary(config: dict, location_root: Path) -> tuple:
    build_plan = build_wflow_build_plan(config, {"location_root": location_root})
    domain_plan = plan_wflow_domain_set(config, {"location_root": location_root})
    wflow_cfg = config.get("wflow", {}) or {}
    domain_set = wflow_cfg.get("domain_set", {}) or {}
    summary = pd.Series(
        {
            "allow_multiple_submodels": domain_set.get("allow_multiple_submodels", False),
            "review_required": build_plan.review_required,
            "domain_status": build_plan.domain_status,
            "reviewed_subbasin_plan_status": domain_plan.status,
            "hydromt_region_kind": build_plan.region_kind,
            "event_catalog_scope": domain_set.get("event_catalog_scope", "shared_across_domain_set"),
            "configured_submodel_count": len(domain_set.get("submodels", []) or []),
            "reviewed_submodel_count": domain_plan.submodel_count,
            "reviewed_handoff_count": domain_plan.handoff_count,
            "domain_set_manifest": wflow_cfg.get("domain_set_manifest", "data/wflow/domain_set.yaml"),
        }
    )
    return build_plan, domain_plan, summary


def subbasins(domain_plan) -> pd.DataFrame:
    if not domain_plan.submodels:
        return pd.DataFrame(
            [{"status": domain_plan.status, "issue": issue} for issue in domain_plan.issues]
        )
    rows = []
    for submodel in domain_plan.submodels:
        outlet_region = submodel.get("outlet_region", submodel["region"])
        outlet_xy = outlet_region.get("subbasin") if isinstance(outlet_region, dict) else None
        outlet_lon, outlet_lat = outlet_xy if outlet_xy else (None, None)
        rows.append(
            {
                "wflow_submodel_id": submodel["wflow_submodel_id"],
                "hydromt_region_kind": submodel["region_kind"],
                "hydromt_region": submodel["region"],
                "handoff_outlet_lon": outlet_lon,
                "handoff_outlet_lat": outlet_lat,
                "sfincs_domain_ids": ", ".join(submodel["sfincs_domain_ids"]),
                "sfincs_handoff_ids": ", ".join(submodel["sfincs_handoff_ids"]),
                "gauge_site_nos": ", ".join(submodel["gauge_site_nos"]),
                "frequency_basis": ", ".join(submodel["frequency_basis"]),
            }
        )
    return pd.DataFrame(rows)


def prepare_wflow_subbasin_fabric(config: dict, location_root: Path, domain_plan) -> tuple:
    wflow = config["wflow"]
    data_sources = config["collection"]["national_hydrography"]
    inputs = exists_table(
        location_root,
        {
            "NHDPlus HR river geometry": data_sources["river_geometry"],
            "NHDPlus HR catchments": data_sources["catchments"],
        },
    )
    if domain_plan.status == "ready" and inputs["exists"].all():
        result = write_wflow_subbasin_fabric_from_nhdplus(config, {"location_root": location_root})
    else:
        subbasin_fabric_path = resolve_location_path(
            location_root,
            wflow["domain_set"].get("subbasin_fabric", "data/wflow/domain_set_subbasins.gpkg"),
        )
        result = {
            "subbasin_fabric": subbasin_fabric_path,
            "subbasin_geometry_files": tuple(sorted(subbasin_fabric_path.with_suffix("").glob("*.geojson"))),
            "diagnostics_csv": resolve_location_path(
                location_root,
                wflow["domain_set"].get(
                    "subbasin_fabric_diagnostics",
                    "data/wflow/readiness/nhdplus_subbasin_fabric.csv",
                ),
            ),
            "submodel_count": 0,
            "catchment_count": 0,
            "statuses": ("missing_inputs_or_review_required",),
        }
    domain_plan = plan_wflow_domain_set(config, {"location_root": location_root})
    summary = pd.Series(
        {
            "subbasin_fabric": str(result["subbasin_fabric"]),
            "subbasin_geometry_files": len(result.get("subbasin_geometry_files", ())),
            "diagnostics_csv": str(result["diagnostics_csv"]),
            "submodel_count": result["submodel_count"],
            "catchment_count": result["catchment_count"],
            "statuses": ", ".join(result["statuses"]),
            "coverage_status": result.get("coverage_status"),
            "coverage_catchment_count": result.get("coverage_catchment_count", 0),
            "evaluation_footprint_within_domain": result.get("evaluation_footprint_within_domain"),
            "evaluation_footprint_uncovered_km2": result.get("evaluation_footprint_uncovered_km2"),
            "power_extent_within_domain": result.get("power_extent_within_domain"),
            "power_extent_uncovered_km2": result.get("power_extent_uncovered_km2"),
            "replanned_status": domain_plan.status,
            "replanned_hydromt_region_kinds": ", ".join(
                sorted({submodel["region_kind"] for submodel in domain_plan.submodels})
            ),
        },
        name="wflow_subbasin_fabric_result",
    )
    return result, inputs, domain_plan, summary
