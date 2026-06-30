from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wflow_runs.notebook import resolve_location_path
from wflow_v2.qa import read_acceptance as read_dynamic_handoff_acceptance
from wflow_v2.qa import validate_event_boundary as _validate_event_boundary
from wflow_v2.qa import validate_handoff_gauge_locations


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


def discharge_source_ids(discharge_nc) -> list[str]:
    with xr.open_dataset(discharge_nc) as ds:
        if "name" in ds:
            return [str(value) for value in ds["name"].values.tolist()]
        return [str(value) for value in ds.get("index", []).values.tolist()]


CFS_TO_CMS = 0.028316846592


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
        from wflow_runs.build_plan import write_wflow_reservoir_readiness

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


def validate_baseflow_against_observed(
    config: dict,
    location_root,
    *,
    zero_rain_discharge_nc,
    streamflow_records_csv=None,
    min_baseflow_fraction: float = 0.25,
) -> pd.DataFrame:
    """Confirm warm-state spin-up left non-dry channels.

    Compares the zero-rain control's per-handoff baseflow to the observed low-flow at the
    Primary Reference Gage, transferred to each crossing by the drainage-area (``uparea``)
    ratio. A near-zero simulated baseflow (flatlined / dry river) fails the gate so spin-up
    can be extended, rather than silently handing SFINCS a dry inflow.
    """
    import geopandas as gpd

    location_root = Path(location_root)
    cfg = (((config.get("inland_coupling", {}) or {}).get("baseflow", {})) or {})
    rows: list[dict] = []
    if not cfg.get("enabled", True):
        return pd.DataFrame([{"check": "baseflow", "status": "disabled", "message": "inland_coupling.baseflow.enabled is false"}])

    reference_gage = str(
        cfg.get("reference_gage")
        or (((config.get("inland_coupling", {}) or {}).get("amplification", {}) or {}).get("primary_reference_gage"))
        or ""
    )
    statistic = str(cfg.get("reference_statistic", "median")).lower()
    if not reference_gage:
        return pd.DataFrame([{"check": "baseflow", "status": "skipped", "message": "no reference_gage configured"}])

    records_csv = Path(streamflow_records_csv) if streamflow_records_csv else (
        location_root / "data/sources/usgs_streamgages/streamflow_records.csv"
    )
    if not records_csv.exists():
        return pd.DataFrame([{"check": "baseflow", "status": "skipped", "message": f"missing {records_csv}"}])
    records = pd.read_csv(records_csv, dtype={"site_no": str})
    site = records[records["site_no"].astype(str).str.zfill(8) == reference_gage.zfill(8)]
    q = pd.to_numeric(site.get("discharge_cfs"), errors="coerce").dropna()
    if q.empty:
        return pd.DataFrame([{"check": "baseflow", "status": "skipped", "message": f"no records for {reference_gage}"}])
    observed_baseflow_cfs = {
        "median": float(q.median()),
        "annual_mean": float(q.mean()),
        "q90": float(q.quantile(0.10)),
    }.get(statistic, float(q.median()))
    observed_baseflow_cms = observed_baseflow_cfs * CFS_TO_CMS

    uparea_by_handoff, reference_uparea = _handoff_upareas(location_root, reference_gage, gpd)
    if not uparea_by_handoff or not reference_uparea:
        return pd.DataFrame([{"check": "baseflow", "status": "skipped", "message": "could not resolve handoff/reference uparea"}])

    with xr.open_dataset(zero_rain_discharge_nc) as opened:
        ds = opened.load()
    names = [str(v) for v in ds["name"].values.tolist()] if "name" in ds else [str(v) for v in ds["index"].values.tolist()]
    sim = np.asarray(ds["discharge"].transpose("index", "time").values, dtype=float)
    for i, handoff_id in enumerate(names):
        upa = uparea_by_handoff.get(handoff_id)
        if not upa:
            continue
        expected_cms = observed_baseflow_cms * (float(upa) / float(reference_uparea))
        simulated_cms = float(np.nanmin(sim[i])) if sim[i].size else 0.0
        ok = simulated_cms >= float(min_baseflow_fraction) * expected_cms
        rows.append(
            {
                "check": "baseflow",
                "sfincs_handoff_id": handoff_id,
                "observed_baseflow_cms": round(expected_cms, 4),
                "simulated_baseflow_cms": round(simulated_cms, 4),
                "status": "passed" if ok else "failed",
                "message": (
                    f"{statistic} ref={observed_baseflow_cfs:.1f} cfs; uparea_ratio={float(upa)/float(reference_uparea):.3f}; "
                    f"min_fraction={min_baseflow_fraction}"
                ),
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        [{"check": "baseflow", "status": "skipped", "message": "no handoff matched uparea map"}]
    )


def _handoff_upareas(location_root: Path, reference_gage: str, gpd):
    """Map sfincs_handoff_id -> Wflow uparea, and the reference gage's uparea."""
    uparea_by_handoff: dict[str, float] = {}
    for path in sorted(location_root.glob("data/sfincs/domains/*/base/gis/wflow_handoff_sources.geojson")):
        gdf = gpd.read_file(path)
        if "sfincs_handoff_id" not in gdf or "uparea" not in gdf:
            continue
        for _, r in gdf.iterrows():
            uparea_by_handoff[str(r["sfincs_handoff_id"])] = float(r["uparea"])
    reference_uparea = None
    for path in sorted(location_root.glob("data/wflow/domain_set_gauges/*_observation_gauges.geojson")):
        gdf = gpd.read_file(path)
        if "site_no" not in gdf or "uparea" not in gdf:
            continue
        match = gdf[gdf["site_no"].astype(str).str.zfill(8) == reference_gage.zfill(8)]
        if not match.empty:
            reference_uparea = float(match.iloc[0]["uparea"])
            break
    return uparea_by_handoff, reference_uparea


def _sfincs_domain_gdf(location_root: Path, sfincs_domains: list[dict], gpd):
    layers = [gpd.read_file(_location_path(location_root, domain["region"])) for domain in sfincs_domains]
    frame = pd.concat(layers, ignore_index=True)
    return gpd.GeoDataFrame(frame, geometry="geometry", crs=layers[0].crs).to_crs(4326)


def _wflow_watershed_gdf(config: dict, location_root: Path, wflow_domain_plan, gpd):
    layers = []
    for submodel in wflow_domain_plan.submodels:
        watershed_path = submodel.get("subbasin_geometry")
        if watershed_path:
            watershed = gpd.read_file(_location_path(location_root, watershed_path)).to_crs(4326)
            watershed["wflow_submodel_id"] = submodel["wflow_submodel_id"]
            layers.append(watershed)
    if not layers:
        watershed_path = config["static_sources"]["wflow_collection_extent"]["watersheds"]
        layers.append(gpd.read_file(_location_path(location_root, watershed_path)).to_crs(4326))
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


def _location_path(location_root: Path, value) -> Path:
    return resolve_location_path(location_root, value)


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


def validate_dynamic_handoff(
    event_discharge_nc,
    *,
    zero_rain_discharge_nc=None,
    expected_source_ids: set[str] | None = None,
    max_zero_peak_fraction: float | None = None,
    max_source_shape_correlation: float = 0.9999,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    report = _validate_event_boundary(
        event_discharge_nc,
        zero_rain_discharge_nc=zero_rain_discharge_nc,
        expected_source_ids=expected_source_ids,
        max_zero_peak_fraction=max_zero_peak_fraction,
        max_shape_correlation=max_source_shape_correlation,
        raise_on_error=False,
    )
    failed = report[report["status"].isin(["failed", "review_required"])]
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.check}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Dynamic Wflow handoff QA failed: {details}")
    return report


def write_dynamic_handoff_acceptance(path, *, event_id: str, discharge_nc, qa_report: pd.DataFrame, metadata: dict | None = None) -> Path:
    path = Path(path)
    accepted = bool(not qa_report["status"].isin(["failed", "review_required"]).any())
    payload = {
        "event_id": str(event_id),
        "status": "accepted" if accepted else "failed",
        "discharge_source": "wflow_dynamic",
        "discharge_nc": str(discharge_nc),
        "checks": qa_report.to_dict(orient="records"),
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
