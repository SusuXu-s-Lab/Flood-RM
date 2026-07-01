from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from paths import resolve_location_path


@dataclass(frozen=True)
class CoupledDomainReview:
    """Geometry review for the Wflow watershed feeding SFINCS coverage."""

    summary: pd.Series
    fig: object
    ax: object
    sfincs_domain_gdf: object
    wflow_watershed_gdf: object
    selected_study_gdf: object
    handoff_plan_gdf: object


@dataclass(frozen=True)
class WflowArtifactInventory:
    """Wflow build artifacts plus native snapped handoff counts."""

    inventory: pd.DataFrame
    native_handoffs: pd.DataFrame
    reservoir_readiness: pd.DataFrame


@dataclass(frozen=True)
class WflowHandoffContract:
    """Final Wflow gauge contract for SFINCS discharge handoff."""

    handoff_contract: pd.DataFrame
    gauge_layers: pd.DataFrame


def coupled_domain_review(
    config: dict,
    location_root,
    *,
    sfincs_domains: list[dict],
    wflow_domain_plan,
    figsize=(9, 7),
) -> CoupledDomainReview:
    """Plot and summarize the pre-build candidate watershed feeding SFINCS coverage."""
    import geopandas as gpd
    import matplotlib.pyplot as plt

    location_root = Path(location_root)
    model_crs = config.get("sfincs", {}).get("model_crs", config.get("project", {}).get("model_crs", "EPSG:32617"))
    fig, ax = plt.subplots(figsize=figsize)

    sfincs_domain_gdf = _sfincs_domain_gdf(location_root, sfincs_domains, gpd)
    sfincs_label = "SFINCS coverage box" if len(sfincs_domain_gdf) == 1 else "SFINCS coverage boxes"
    sfincs_coverage = sfincs_domain_gdf.geometry.union_all()

    wflow_watershed_gdf = _wflow_watershed_gdf(config, location_root, wflow_domain_plan, gpd)
    wflow_watershed_gdf.boundary.plot(ax=ax, color="#2b6cb0", linewidth=1.4, label="Wflow boundary-handoff watershed")

    selected_study, selected_ids = _selected_study_footprint(location_root, sfincs_domains, sfincs_coverage, gpd)
    selected_study_geom = selected_study.geometry.union_all() if not selected_study.empty else None
    if not selected_study.empty:
        selected_study.boundary.plot(ax=ax, color="black", linewidth=1.0, label="selected SMART-DS footprint")

    sfincs_domain_gdf.boundary.plot(ax=ax, color="red", linewidth=1.4, linestyle="--", label=sfincs_label)
    handoff_plan_gdf = _handoff_plan_gdf(wflow_domain_plan, gpd)
    if not handoff_plan_gdf.empty:
        handoff_plan_gdf.plot(
            ax=ax,
            marker="D",
            color="crimson",
            markersize=35,
            label="candidate handoff point (pre-build)",
        )

    wflow_watershed = wflow_watershed_gdf.geometry.union_all()
    max_handoff_distance = _max_boundary_distance(handoff_plan_gdf, sfincs_domain_gdf, model_crs)
    summary_items = {
        "wflow_watershed_features": int(len(wflow_watershed_gdf)),
        "candidate_boundary_handoff_count": int(len(handoff_plan_gdf)),
        "max_candidate_handoff_distance_from_sfincs_boundary_m": (
            round(max_handoff_distance, 3) if max_handoff_distance is not None else "no candidate handoffs"
        ),
    }
    if selected_ids:
        summary_items["selected_smart_ds_subregions"] = ", ".join(sorted(selected_ids))
        summary_items["sfincs_coverage_covers_selected_footprint"] = _covers(sfincs_coverage, selected_study_geom, model_crs)
        summary_items["wflow_watershed_covers_selected_footprint"] = _covers(wflow_watershed, selected_study_geom, model_crs)
    else:
        summary_items["selected_smart_ds_subregions"] = "spatial_intersection_fallback"
    summary_items["wflow_watershed_covers_sfincs_coverage"] = _covers(wflow_watershed, sfincs_coverage, model_crs)

    ax.set_title(f"Candidate Wflow watershed feeding selected {sfincs_label.lower()}")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="best")
    return CoupledDomainReview(
        summary=pd.Series(summary_items, name="candidate_domain_hydrologic_boundary_check"),
        fig=fig,
        ax=ax,
        sfincs_domain_gdf=sfincs_domain_gdf,
        wflow_watershed_gdf=wflow_watershed_gdf,
        selected_study_gdf=selected_study,
        handoff_plan_gdf=handoff_plan_gdf,
    )


def wflow_artifact_inventory(
    config: dict,
    location_root,
    *,
    selected_submodels: list[dict],
    wflow_models: dict,
    wflow_base_root,
    repo_root,
    load_staticmap_values: bool = False,
) -> WflowArtifactInventory:
    """Inventory Wflow static maps/geometries and native snapped handoff gauges."""
    rows = []
    native_rows = []
    wflow_base_root = Path(wflow_base_root)
    repo_root = Path(repo_root)
    for submodel in selected_submodels:
        submodel_id = str(submodel["wflow_submodel_id"])
        wf = wflow_models[submodel_id]
        staticmaps_path = wflow_base_root / submodel_id / "staticmaps.nc"
        for name, data_array in wf.staticmaps.data.data_vars.items():
            rows.append(_staticmap_row(submodel_id, name, data_array, staticmaps_path, repo_root, load_staticmap_values))
        for name, geom in wf.geoms.data.items():
            geom_path = wflow_base_root / submodel_id / "staticgeoms" / f"{name}.geojson"
            rows.append(
                {
                    "wflow_submodel_id": submodel_id,
                    "artifact": "staticgeoms",
                    "name": name,
                    "item_kind": "features",
                    "item_count": int(len(geom)),
                    "artifact_path": _relative_path(geom_path, repo_root),
                    "exists_on_disk": geom_path.exists(),
                }
            )

        planned = submodel.get("sfincs_handoff_ids", [])
        native = wf.geoms.data.get("gauges_sfincs")
        native_ids = [] if native is None else sorted(native["name"].astype(str).tolist())
        native_rows.append(
            {
                "wflow_submodel_id": submodel_id,
                "planned_boundary_crossings": len(planned),
                "native_snapped_wflow_gauges": len(native_ids),
                "native_gauge_ids": ", ".join(native_ids),
            }
        )

    inventory = pd.DataFrame(rows)
    missing = sorted(inventory.loc[~inventory["exists_on_disk"], "artifact_path"].astype(str).unique()) if not inventory.empty else []
    if missing:
        raise FileNotFoundError(
            "Wflow build did not write required artifacts; rerun Step 5 to completion before continuing: "
            + ", ".join(missing)
        )

    reservoir_readiness = pd.DataFrame()
    if _reservoirs_enabled(config):
        from wflow_runs.reservoirs import write_wflow_reservoir_readiness

        reservoir_readiness = write_wflow_reservoir_readiness(config, location_root, raise_on_error=False)
        if not reservoir_readiness.empty and reservoir_readiness["status"].isin(["failed"]).any():
            raise RuntimeError("Wflow reservoir readiness failed; rebuild the Wflow base with native reservoirs before continuing.")
    return WflowArtifactInventory(inventory, pd.DataFrame(native_rows), reservoir_readiness)


def wflow_handoff_contract(
    config: dict,
    location_root,
    *,
    wflow_models: dict,
    handoff_gdf,
    wflow_base_root,
) -> WflowHandoffContract:
    """Verify each SFINCS source is represented by one nearby Wflow gauge."""
    import geopandas as gpd

    model_crs = config.get("sfincs", {}).get("model_crs", config.get("project", {}).get("model_crs", "EPSG:32617"))
    max_snap_m = float(config.get("sfincs", {}).get("grid_resolution_m", 100))
    gauge_rows = []
    contract_rows = []
    for submodel_id, wf in wflow_models.items():
        submodel_id = str(submodel_id)
        for name, geom in wf.geoms.data.items():
            if name.startswith("gauges") or name.startswith("subcatchment"):
                gauge_rows.append({"wflow_submodel_id": submodel_id, "geometry_layer": name, "feature_count": len(geom)})

        sfincs_gauges = wf.geoms.data.get("gauges_sfincs")
        if sfincs_gauges is None or sfincs_gauges.empty:
            raise ValueError(f"{submodel_id} has no gauges_sfincs layer for SFINCS discharge handoff.")

        sources = handoff_gdf[handoff_gdf["wflow_submodel_id"].astype(str).eq(submodel_id)].to_crs(model_crs)
        if sources.empty:
            raise ValueError(f"{submodel_id} has no SFINCS handoff sources to couple back to Wflow.")

        input_distances = _validate_input_gauges(location_root, wflow_base_root, submodel_id, sources, model_crs, gpd)
        gauges = sfincs_gauges.to_crs(model_crs).copy()
        _require_matching_ids(submodel_id, sources, gauges, "gauges_sfincs", "Wflow")
        for _, source in sources.iterrows():
            gauge_id = str(source["sfincs_handoff_id"])
            matched = _match_gauge(gauges, gauge_id)
            distances = matched.geometry.distance(source.geometry)
            nearest_index = distances.idxmin()
            nearest = matched.loc[nearest_index]
            row = {
                "wflow_submodel_id": submodel_id,
                "sfincs_handoff_id": source["sfincs_handoff_id"],
                "wflow_gauge_index": nearest.get("index"),
                "wflow_gauge_handoff_id": nearest.get("sfincs_handoff_id", nearest.get("name")),
                "source_to_wflow_gauge_m": round(float(distances.loc[nearest_index]), 3),
            }
            if input_distances is not None:
                row["wflow_input_to_source_m"] = input_distances[gauge_id]
            contract_rows.append(row)

    contract = pd.DataFrame(contract_rows)
    if not contract.empty:
        contract["source_count_per_wflow_gauge"] = contract.groupby(
            ["wflow_submodel_id", "wflow_gauge_index"]
        )["sfincs_handoff_id"].transform("count")
        bad_snap = contract[contract["source_to_wflow_gauge_m"].gt(max_snap_m)]
        if not bad_snap.empty:
            raise ValueError(
                "SFINCS handoff sources are farther than one SFINCS grid cell from their nearest Wflow gauges: "
                + ", ".join(bad_snap["sfincs_handoff_id"].astype(str))
            )
    return WflowHandoffContract(contract, pd.DataFrame(gauge_rows))


def _sfincs_domain_gdf(location_root: Path, sfincs_domains: list[dict], gpd):
    layers = [gpd.read_file(resolve_location_path(location_root, domain["region"])) for domain in sfincs_domains]
    frame = pd.concat(layers, ignore_index=True)
    return gpd.GeoDataFrame(frame, geometry="geometry", crs=layers[0].crs).to_crs(4326)


def _wflow_watershed_gdf(config: dict, location_root: Path, wflow_domain_plan, gpd):
    layers = []
    for submodel in wflow_domain_plan.submodels:
        watershed_path = submodel.get("subbasin_geometry")
        if watershed_path:
            watershed = gpd.read_file(resolve_location_path(location_root, watershed_path)).to_crs(4326)
            watershed["wflow_submodel_id"] = submodel["wflow_submodel_id"]
            layers.append(watershed)
    if not layers:
        watershed_path = config["static_sources"]["wflow_collection_extent"]["watersheds"]
        layers.append(gpd.read_file(resolve_location_path(location_root, watershed_path)).to_crs(4326))
    return gpd.GeoDataFrame(pd.concat(layers, ignore_index=True), geometry="geometry", crs=layers[0].crs).to_crs(4326)


def _selected_study_footprint(location_root: Path, sfincs_domains: list[dict], sfincs_coverage, gpd):
    study = gpd.read_file(location_root / "data/static/aoi/evaluation_footprint.geojson").to_crs(4326)
    selected_ids = {
        str(domain.get("exposure_subregion_id")).strip()
        for domain in sfincs_domains
        if str(domain.get("exposure_subregion_id", "")).strip()
    }
    if selected_ids and "subregion_id" in study.columns:
        selected = study[study["subregion_id"].astype(str).isin(selected_ids)].copy()
    else:
        components = study.explode(index_parts=False).reset_index(drop=True)
        selected = components[components.intersects(sfincs_coverage)].copy()
    if selected.empty:
        raise RuntimeError("No SMART-DS footprint geometry matched the selected SFINCS domains")
    return selected, selected_ids


def _handoff_plan_gdf(wflow_domain_plan, gpd):
    records = []
    for submodel in wflow_domain_plan.submodels:
        for point in submodel.get("handoff_points", []):
            records.append(
                {
                    "sfincs_handoff_id": point["sfincs_handoff_id"],
                    "wflow_submodel_id": submodel["wflow_submodel_id"],
                    "sfincs_domain_id": point.get("sfincs_domain_id"),
                    "uparea_km2": point.get("uparea_km2"),
                    "geometry": gpd.points_from_xy([point["lon"]], [point["lat"]])[0],
                }
            )
    if not records:
        return gpd.GeoDataFrame(
            columns=["sfincs_handoff_id", "wflow_submodel_id", "sfincs_domain_id", "uparea_km2", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def _max_boundary_distance(handoff_gdf, sfincs_domain_gdf, model_crs: str) -> float | None:
    if handoff_gdf.empty:
        return None
    boundary = sfincs_domain_gdf.to_crs(model_crs).geometry.union_all().boundary
    distances = handoff_gdf.to_crs(model_crs).geometry.distance(boundary)
    return float(distances.max())


def _covers(outer_geom, inner_geom, model_crs: str, tol_m2: float = 1.0) -> bool:
    if outer_geom is None or inner_geom is None:
        return False
    import geopandas as gpd

    outer = gpd.GeoSeries([outer_geom], crs=4326).to_crs(model_crs).iloc[0]
    inner = gpd.GeoSeries([inner_geom], crs=4326).to_crs(model_crs).iloc[0]
    return bool(inner.difference(outer).area <= tol_m2)


def _staticmap_row(submodel_id: str, name: str, data_array, staticmaps_path: Path, repo_root: Path, load_values: bool) -> dict:
    finite_cells = "not_loaded"
    missing_cells = "not_loaded"
    if load_values:
        values = data_array.values
        finite_cells = int(pd.notna(values).sum())
        missing_cells = int(pd.isna(values).sum())
    return {
        "wflow_submodel_id": submodel_id,
        "artifact": "staticmaps",
        "name": name,
        "item_kind": "cells",
        "item_count": int(data_array.size),
        "finite_cells": finite_cells,
        "missing_cells": missing_cells,
        "dimensions": " x ".join(f"{dim}={size}" for dim, size in data_array.sizes.items()),
        "artifact_path": _relative_path(staticmaps_path, repo_root),
        "exists_on_disk": staticmaps_path.exists(),
    }


def _validate_input_gauges(location_root, wflow_base_root, submodel_id: str, sources, model_crs: str, gpd):
    input_path = Path(wflow_base_root).parent / "domain_set_gauges" / f"{submodel_id}_sfincs_gauges.geojson"
    if not input_path.exists():
        return None
    input_gauges = gpd.read_file(input_path).to_crs(model_crs)
    _require_matching_ids(submodel_id, sources, input_gauges, "Wflow SFINCS gauge input", "input")
    distances = {}
    for _, source in sources.iterrows():
        gauge_id = str(source["sfincs_handoff_id"])
        matched = _match_gauge(input_gauges, gauge_id)
        distances[gauge_id] = round(float(matched.geometry.distance(source.geometry).min()), 3)
    stale = {gauge_id: distance for gauge_id, distance in distances.items() if distance > 1.0}
    if stale:
        preview = ", ".join(f"{gauge_id}={distance:.1f} m" for gauge_id, distance in sorted(stale.items())[:8])
        raise RuntimeError(
            f"{submodel_id} Wflow SFINCS gauge input coordinates are stale relative to current SFINCS handoff sources "
            f"({preview}). Set rerun=True, rerun Step 11, then rerun Steps 12-13."
        )
    return distances


def _require_matching_ids(submodel_id: str, sources, gauges, label: str, target: str) -> None:
    source_ids = set(sources["sfincs_handoff_id"].astype(str))
    gauge_ids = _gauge_ids(gauges)
    missing = sorted(source_ids - gauge_ids)
    stale = sorted(gauge_ids - source_ids)
    if missing or stale:
        raise RuntimeError(
            f"{submodel_id} {label} is stale relative to current SFINCS handoff sources. "
            f"Missing in {target}: {missing or 'none'}; stale in {target}: {stale or 'none'}. "
            "Set rerun=True, rerun Step 11, then rerun Steps 12-13."
        )


def _gauge_ids(gauges) -> set[str]:
    if "sfincs_handoff_id" in gauges:
        return set(gauges["sfincs_handoff_id"].astype(str))
    return set(gauges["name"].astype(str))


def _match_gauge(gauges, gauge_id: str):
    if "sfincs_handoff_id" in gauges:
        return gauges[gauges["sfincs_handoff_id"].astype(str).eq(gauge_id)]
    return gauges[gauges["name"].astype(str).eq(gauge_id)]


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _reservoirs_enabled(config: dict) -> bool:
    return bool(
        ((config.get("collection", {}) or {}).get("national_hydrography", {}) or {})
        .get("reservoirs", {})
        .get("enabled", False)
    )


__all__ = [
    "CoupledDomainReview",
    "WflowArtifactInventory",
    "WflowHandoffContract",
    "coupled_domain_review",
    "wflow_artifact_inventory",
    "wflow_handoff_contract",
]
