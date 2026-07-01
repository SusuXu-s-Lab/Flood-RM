from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from paths import location_root_from_paths, resolve_location_path


def write_wflow_subbasin_fabric_from_nhdplus(config, paths) -> dict:
    """Write reviewed Wflow subbasin polygons from NHDPlus HR catchments."""
    from coupling.wflow_domain_set import plan_wflow_domain_set

    location_root = location_root_from_paths(paths)
    plan = plan_wflow_domain_set(config, paths)
    if plan.status != "ready":
        raise ValueError(f"Wflow Domain Set plan is not ready: {plan.status}")

    collection = config.get("collection", {}).get("national_hydrography", {})
    wflow = config.get("wflow", {})
    catchments_path = resolve_location_path(
        location_root,
        collection.get("catchments", "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"),
    )
    rivers_path = resolve_location_path(
        location_root,
        collection.get("river_geometry", "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"),
    )
    output_path = resolve_location_path(
        location_root,
        wflow.get("domain_set", {}).get("subbasin_fabric", "data/wflow/domain_set_subbasins.gpkg"),
    )
    diagnostics_path = resolve_location_path(
        location_root,
        wflow.get("domain_set", {}).get("subbasin_fabric_diagnostics", "data/wflow/readiness/nhdplus_subbasin_fabric.csv"),
    )
    if not catchments_path.exists():
        raise FileNotFoundError(catchments_path)
    if not rivers_path.exists():
        raise FileNotFoundError(rivers_path)

    catchments = gpd.read_file(catchments_path).to_crs("EPSG:4326")
    rivers = gpd.read_file(rivers_path).to_crs("EPSG:4326")
    if catchments.empty:
        raise ValueError(f"NHDPlus HR catchments artifact has no features: {catchments_path}")
    if rivers.empty:
        raise ValueError(f"NHDPlus HR river geometry artifact has no features: {rivers_path}")

    handoff = wflow.get("handoff", {})
    rows = []
    diagnostics = []
    for submodel in plan.submodels:
        selected_catchments, outlet_matches, method = _select_submodel_upstream_nhdplus_catchments(
            submodel,
            rivers,
            catchments,
        )
        geometry = selected_catchments.geometry.union_all()
        nhdplus_area_km2 = _equal_area_km2(geometry)
        reviewed_area_km2 = _reviewed_submodel_handoff_area_km2(submodel)
        area_difference_pct = _area_difference_pct(nhdplus_area_km2, reviewed_area_km2)
        area_match_status = _area_match_status(area_difference_pct)
        review_status = (
            "review_required_nhdplus_upstream"
            if "routed_upstream_catchments" in method
            else "review_required_nearest_catchment_only"
        )
        representative = outlet_matches[0]
        rows.append(
            {
                "wflow_submodel_id": submodel["wflow_submodel_id"],
                "region_role": "handoff_drainage",
                "sfincs_handoff_ids": ",".join(submodel["sfincs_handoff_ids"]),
                "sfincs_boundary_id": ",".join(submodel["sfincs_handoff_ids"]),
                "sfincs_boundary_type": "discharge",
                "sfincs_forcing_target": handoff.get("target", "sfincs_discharge_forcing"),
                "wflow_source_variable": handoff.get("source_variable", "river_q"),
                "wflow_source_standard_name": handoff.get(
                    "source_standard_name",
                    "river_water__volume_flow_rate",
                ),
                "gauge_site_nos": ",".join(submodel["gauge_site_nos"]),
                "outlet_lon": float(representative["lon"]),
                "outlet_lat": float(representative["lat"]),
                "catchment_count": int(len(selected_catchments)),
                "nhdplus_area_km2": round(nhdplus_area_km2, 3),
                "reviewed_drainage_area_km2": round(float(reviewed_area_km2), 3) if reviewed_area_km2 is not None else float("nan"),
                "area_difference_pct": round(area_difference_pct, 3) if area_difference_pct is not None else float("nan"),
                "area_match_status": area_match_status,
                "aggregation_method": method,
                "review_status": review_status,
                "source": "USGS NHDPlus HR NHDPlusCatchment",
                "geometry": geometry,
            }
        )
        for match in outlet_matches:
            diagnostics.append(
                {
                    "wflow_submodel_id": submodel["wflow_submodel_id"],
                    "sfincs_handoff_id": match["sfincs_handoff_id"],
                    "outlet_lon": float(match["lon"]),
                    "outlet_lat": float(match["lat"]),
                    "matched_river_index": int(match["river_index"]),
                    "matched_catchment_index": int(match["catchment_index"]),
                    "river_snap_distance_m": float(match["river_distance_m"]),
                    "catchment_match_distance_m": float(match["catchment_distance_m"]),
                    "catchment_count": int(len(selected_catchments)),
                    "nhdplus_area_km2": round(nhdplus_area_km2, 3),
                    "reviewed_drainage_area_km2": round(float(reviewed_area_km2), 3) if reviewed_area_km2 is not None else float("nan"),
                    "area_difference_pct": round(area_difference_pct, 3) if area_difference_pct is not None else float("nan"),
                    "area_match_status": area_match_status,
                    "aggregation_method": method,
                    "review_status": review_status,
                }
            )

    handoff_rows = list(rows)
    submodel_count = len(handoff_rows)
    handoff_catchment_count = int(sum(int(row["catchment_count"]) for row in handoff_rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    fabric = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    fabric.to_file(output_path, driver="GPKG")
    fabric_dir = _subbasin_fabric_directory(output_path)
    fabric_dir.mkdir(parents=True, exist_ok=True)
    member_ids = {str(value) for value in fabric["wflow_submodel_id"]}
    for stale_path in fabric_dir.glob("*.geojson"):
        if stale_path.stem not in member_ids:
            stale_path.unlink()
    subbasin_geometry_files = []
    coverage_geometry_file = None
    for _, row in fabric.iterrows():
        member_id = str(row["wflow_submodel_id"])
        geojson_path = fabric_dir / f"{member_id}.geojson"
        gpd.GeoDataFrame([row], geometry="geometry", crs=fabric.crs).to_file(geojson_path, driver="GeoJSON")
        if row.get("region_role") == "evaluation_coverage":
            coverage_geometry_file = geojson_path
        else:
            subbasin_geometry_files.append(geojson_path)
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)
    return {
        "subbasin_fabric": output_path,
        "subbasin_geometry_files": tuple(subbasin_geometry_files),
        "diagnostics_csv": diagnostics_path,
        "submodel_count": submodel_count,
        "catchment_count": handoff_catchment_count,
        "area_mismatch_count": int(sum(1 for row in handoff_rows if row.get("area_match_status") == "review_required_area_mismatch")),
        "area_mismatch_submodels": tuple(
            str(row["wflow_submodel_id"])
            for row in handoff_rows
            if row.get("area_match_status") == "review_required_area_mismatch"
        ),
        "statuses": tuple(sorted(fabric["review_status"].unique())),
        "coverage_region": str(coverage_geometry_file) if coverage_geometry_file else None,
        "coverage_catchment_count": 0,
        "coverage_status": "handoff_watershed_only",
    }


def _select_submodel_upstream_nhdplus_catchments(submodel: dict, rivers, catchments):
    selections = []
    matches = []
    methods = []
    for handoff in _submodel_handoff_points(submodel):
        outlet = gpd.GeoDataFrame(
            {"_outlet": [handoff["sfincs_handoff_id"]]},
            geometry=[Point(float(handoff["lon"]), float(handoff["lat"]))],
            crs="EPSG:4326",
        )
        match = _match_nhdplus_outlet(outlet, rivers, catchments)
        selected, method = _select_upstream_nhdplus_catchments(
            rivers,
            catchments,
            match["river_index"],
            match["catchment_index"],
        )
        selections.append(selected)
        methods.append(method)
        matches.append({**handoff, **match})
    if not selections:
        raise ValueError(f"Wflow Submodel {submodel.get('wflow_submodel_id')} has no handoff outlet points")
    selected_catchments = gpd.GeoDataFrame(
        pd.concat(selections, ignore_index=True),
        geometry="geometry",
        crs=catchments.crs,
    )
    id_col = _find_column(selected_catchments, ("featureid", "nhdplusid", "comid", "gridcode"))
    if id_col:
        selected_catchments = selected_catchments.drop_duplicates(subset=[id_col]).copy()
    else:
        selected_catchments = selected_catchments.drop_duplicates(subset=["geometry"]).copy()
    method = methods[0] if len(set(methods)) == 1 else "union_" + "_and_".join(sorted(set(methods)))
    return selected_catchments, matches, method


def _submodel_handoff_points(submodel: dict) -> list[dict]:
    points = submodel.get("handoff_points")
    if points:
        return [
            {
                "sfincs_handoff_id": str(point["sfincs_handoff_id"]),
                "lon": float(point["lon"]),
                "lat": float(point["lat"]),
                "uparea_km2": float(point["uparea_km2"]) if point.get("uparea_km2") is not None else np.nan,
            }
            for point in points
        ]

    outlet_xy = (submodel.get("outlet_region") or submodel.get("region") or {}).get("subbasin")
    if not outlet_xy:
        return []
    handoff_id = next(
        (str(value) for value in submodel.get("sfincs_handoff_ids", ()) if value),
        str(submodel.get("wflow_submodel_id", "")),
    )
    uparea = (submodel.get("region") or {}).get("uparea")
    return [
        {
            "sfincs_handoff_id": handoff_id,
            "lon": float(outlet_xy[0]),
            "lat": float(outlet_xy[1]),
            "uparea_km2": float(uparea) if uparea is not None else np.nan,
        }
    ]


def _reviewed_submodel_handoff_area_km2(submodel: dict) -> float | None:
    points = _submodel_handoff_points(submodel)
    areas = [
        float(point["uparea_km2"])
        for point in points
        if point.get("uparea_km2") is not None and not pd.isna(point.get("uparea_km2"))
    ]
    if not areas:
        return None
    return float(sum(areas)) if len(areas) > 1 else float(areas[0])


def _area_difference_pct(nhdplus_area_km2, reviewed_area_km2) -> float | None:
    if reviewed_area_km2 in (None, "") or pd.isna(reviewed_area_km2):
        return None
    reviewed_area_km2 = float(reviewed_area_km2)
    if reviewed_area_km2 <= 0:
        return None
    return abs(float(nhdplus_area_km2) - reviewed_area_km2) / reviewed_area_km2 * 100.0


def _area_match_status(area_difference_pct: float | None) -> str:
    if area_difference_pct is None:
        return "missing_reviewed_area"
    if area_difference_pct > 50.0:
        return "review_required_area_mismatch"
    return "within_review_tolerance"


def _subbasin_fabric_directory(output_path: Path) -> Path:
    return output_path.with_suffix("")


def _match_nhdplus_outlet(outlet, rivers, catchments):
    outlet_m = outlet.to_crs("EPSG:5070")
    rivers_m = rivers.to_crs("EPSG:5070")
    catchments_m = catchments.to_crs("EPSG:5070")
    point = outlet_m.geometry.iloc[0]

    river_distances = rivers_m.geometry.distance(point)
    river_index = int(river_distances.idxmin())
    containing = catchments_m[catchments_m.geometry.contains(point) | catchments_m.geometry.touches(point)]
    if containing.empty:
        catchment_distances = catchments_m.geometry.distance(point)
        catchment_index = int(catchment_distances.idxmin())
        catchment_distance = float(catchment_distances.loc[catchment_index])
    else:
        catchment_index = int(containing.index[0])
        catchment_distance = 0.0
    return {
        "river_index": river_index,
        "catchment_index": catchment_index,
        "river_distance_m": float(river_distances.loc[river_index]),
        "catchment_distance_m": catchment_distance,
    }


def _select_upstream_nhdplus_catchments(rivers, catchments, river_index, catchment_index):
    hydroseq_col = _find_column(rivers, ("hydroseq",))
    downstream_col = _find_column(rivers, ("dnhydroseq", "dn_hydroseq", "tohydroseq", "to_hydroseq"))
    river_id_col = _find_column(rivers, ("nhdplusid", "featureid", "comid"))
    catchment_id_col = _find_column(catchments, ("featureid", "nhdplusid", "comid"))
    if hydroseq_col and downstream_col and river_id_col and catchment_id_col:
        selected_hydroseq = _upstream_hydroseq_values(rivers, river_index, hydroseq_col, downstream_col)
        selected_flowline_ids = set(
            pd.to_numeric(
                rivers.loc[rivers[hydroseq_col].isin(selected_hydroseq), river_id_col],
                errors="coerce",
            ).dropna().astype("int64")
        )
        catchment_ids = pd.to_numeric(catchments[catchment_id_col], errors="coerce")
        selected = catchments[catchment_ids.isin(selected_flowline_ids)].copy()
        if not selected.empty:
            return selected, "routed_upstream_catchments"
    return catchments.loc[[catchment_index]].copy(), "nearest_or_containing_catchment"


def _upstream_hydroseq_values(rivers, outlet_index, hydroseq_col, downstream_col):
    hydroseq = pd.to_numeric(rivers[hydroseq_col], errors="coerce")
    downstream = pd.to_numeric(rivers[downstream_col], errors="coerce")
    outlet_value = hydroseq.loc[outlet_index]
    if pd.isna(outlet_value):
        return set()
    selected = {int(outlet_value)}
    changed = True
    while changed:
        changed = False
        upstream = set(hydroseq[downstream.isin(selected)].dropna().astype("int64"))
        new_values = upstream - selected
        if new_values:
            selected |= new_values
            changed = True
    return selected


def _equal_area_km2(geometry) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    return float(gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs("EPSG:5070").area.iloc[0]) / 1.0e6


def _find_column(frame, candidates):
    columns = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        column = columns.get(str(candidate).lower())
        if column is not None:
            return column
    return None
