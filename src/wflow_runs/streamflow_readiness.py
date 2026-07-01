from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

from event_streamflow import streamflow_member_metadata
from paths import resolve_location_path


WFLOW_EXTERNAL_RIVER_INFLOW_VAR = "river_inflow"


def validate_wflow_streamflow_realization(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    event_model_root=None,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """Validate rainfall-driven Wflow event readiness.

    Discharge is the Wflow response, not an injected streamflow member. This check keeps
    the current ADR-0016 contract explicit: an event must have a rainfall member and a
    Same-Frequency Amplification/baseflow reference gage, and generated event inputs
    should not contain legacy external river inflow.
    """
    location_root = Path(location_root)
    row = event_catalog_row(location_root, event_id, catalog_path)
    rows = [
        _catalog_rainfall_row(row),
        _amplification_reference_row(config),
    ]
    if event_model_root is not None:
        rows.extend(_event_model_rainfall_forcing_rows(Path(event_model_root)))
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed"])]
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.check}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Inland rainfall-driven event is not ready for Wflow generation: {details}")
    return report


def wflow_streamflow_gage_overlap(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    submodel_ids: list[str] | None = None,
) -> dict:
    """Describe whether an event's streamflow member overlaps reviewed Wflow gauges."""
    import geopandas as gpd

    location_root = Path(location_root)
    row = event_catalog_row(location_root, event_id, catalog_path)
    member = streamflow_member_metadata(config, location_root, row)
    member_sites = {str(site) for site in member["site_nos"]}
    submodel_ids = submodel_ids or active_wflow_submodel_ids(config, location_root)
    reviewed_sites: set[str] = set()
    gauge_paths: list[str] = []
    missing_paths: list[str] = []
    for submodel_id in submodel_ids:
        gauges_path = observation_gauges_path(config, location_root, submodel_id)
        if not gauges_path.exists():
            missing_paths.append(str(gauges_path))
            continue
        gauge_paths.append(str(gauges_path))
        gauges = gpd.read_file(gauges_path)
        if "site_no" not in gauges:
            raise ValueError(f"{gauges_path} lacks site_no column for streamflow realization")
        reviewed_sites.update(gauges["site_no"].astype(str))

    overlap = sorted(member_sites & reviewed_sites)
    compatible = bool(overlap)
    if compatible:
        message = (
            f"streamflow member {member['member_id']} overlaps reviewed Wflow gauges: "
            + ", ".join(overlap)
        )
    else:
        message = (
            f"streamflow member {member['member_id']} sites do not overlap reviewed Wflow observation gauges "
            f"for active submodels {submodel_ids or 'none'}."
        )
        if missing_paths:
            message += " Missing gauge files: " + ", ".join(missing_paths)
    return {
        "event_id": str(event_id),
        "member_id": member["member_id"],
        "member_sites": sorted(member_sites),
        "submodel_ids": list(submodel_ids),
        "reviewed_site_count": len(reviewed_sites),
        "overlap_site_nos": overlap,
        "compatible": compatible,
        "gauge_paths": gauge_paths,
        "message": message,
    }


def observation_gauges_path(config: dict, location_root: Path, submodel_id: str) -> Path:
    root = (
        ((config.get("wflow", {}) or {}).get("gauges", {}) or {}).get("root")
        or "data/wflow/domain_set_gauges"
    )
    root = resolve_location_path(location_root, root)
    return root / f"{submodel_id}_observation_gauges.geojson"


def active_wflow_submodel_ids(config: dict, location_root: Path) -> list[str]:
    domain_set = ((config.get("wflow", {}) or {}).get("domain_set", {}) or {})
    configured = [
        str(item["wflow_submodel_id"])
        for item in domain_set.get("submodels", []) or []
        if item.get("wflow_submodel_id")
    ]
    if configured:
        return configured

    manifest_path = resolve_location_path(
        location_root,
        (config.get("wflow", {}) or {}).get("domain_set_manifest", "data/wflow/domain_set.yaml"),
    )
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        manifested = [
            str(item["wflow_submodel_id"])
            for item in manifest.get("submodels", []) or []
            if item.get("wflow_submodel_id")
        ]
        if manifested:
            return manifested

    gauges_root = (
        ((config.get("wflow", {}) or {}).get("gauges", {}) or {}).get("root")
        or "data/wflow/domain_set_gauges"
    )
    gauges_root = resolve_location_path(location_root, gauges_root)
    suffix = "_observation_gauges.geojson"
    return sorted(path.name[: -len(suffix)] for path in gauges_root.glob(f"*{suffix}"))


def event_catalog_row(location_root: Path, event_id: str, catalog_path):
    catalog_path = (
        Path(catalog_path)
        if catalog_path
        else resolve_location_path(location_root, "data/event_catalog/catalog/probability_catalog.csv")
    )
    if not catalog_path.is_absolute():
        catalog_path = location_root / catalog_path
    catalog = pd.read_csv(catalog_path)
    catalog["event_id"] = catalog["event_id"].astype(str)
    match = catalog[catalog["event_id"] == str(event_id)]
    if match.empty:
        raise ValueError(f"event_id {event_id!r} not in {catalog_path}")
    return match.iloc[0]


def _catalog_rainfall_row(row: pd.Series) -> dict:
    missing = [
        key
        for key in ("rainfall_member_id", "rainfall_member_file")
        if row.get(key) is None or pd.isna(row.get(key)) or str(row.get(key)).strip() == ""
    ]
    if missing:
        return {"check": "catalog_rainfall_member", "status": "failed", "message": "missing " + ", ".join(missing)}
    return {
        "check": "catalog_rainfall_member",
        "status": "passed",
        "message": f"member={row.get('rainfall_member_id')}; scale={row.get('rainfall_scale_factor')}",
    }


def _amplification_reference_row(config: dict) -> dict:
    gage = (((config.get("inland_coupling", {}) or {}).get("amplification", {}) or {}).get("primary_reference_gage"))
    if not gage:
        return {
            "check": "amplification_reference_gage",
            "status": "review_required",
            "message": "inland_coupling.amplification.primary_reference_gage unset (single-K/baseflow validation anchor)",
        }
    return {"check": "amplification_reference_gage", "status": "passed", "message": f"primary_reference_gage={gage}"}


def _event_model_rainfall_forcing_rows(event_model_root: Path) -> list[dict]:
    forcing_path = event_model_root / "inmaps-event.nc"
    rows: list[dict] = []
    if not forcing_path.exists():
        rows.append({"check": "wflow_event_precip_forcing", "status": "failed", "message": f"missing {forcing_path}"})
        return rows
    with xr.open_dataset(forcing_path) as ds:
        has_precip = "precip" in ds
        has_inflow = WFLOW_EXTERNAL_RIVER_INFLOW_VAR in ds
    rows.append(
        {
            "check": "wflow_event_precip_forcing",
            "status": "passed" if has_precip else "failed",
            "message": "precip present" if has_precip else "precip missing from inmaps-event.nc",
        }
    )
    rows.append(
        {
            "check": "wflow_no_external_inflow",
            "status": "passed" if not has_inflow else "review_required",
            "message": (
                "no external river_inflow (rainfall-driven)"
                if not has_inflow
                else "legacy external river_inflow present - rainfall-runoff double-count risk"
            ),
        }
    )
    return rows
