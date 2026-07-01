from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from paths import resolve_location_path
from wflow_runs.domain import configured_or_manifest_submodels
from coupling.handoff_sources import read_stream_boundary_handoff_location_artifacts


def handoff_source_ids_for_submodel(sources, submodel_id: str) -> set[str]:
    """Return reviewed SFINCS handoff IDs assigned to one Wflow submodel."""
    if sources is None or sources.empty or "sfincs_handoff_id" not in sources:
        return set()
    submodel_sources = sources
    if "wflow_submodel_id" in submodel_sources:
        submodel_sources = submodel_sources[
            submodel_sources["wflow_submodel_id"].astype(str) == str(submodel_id)
        ].copy()
    return set(submodel_sources["sfincs_handoff_id"].dropna().astype(str))


def handoff_artifact_row(
    submodel_id: str,
    check: str,
    path: str | Path | None,
    expected: set[str],
    actual: set[str],
) -> dict[str, str]:
    """Build the standard Wflow-SFINCS handoff artifact QA row."""
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    status = "passed" if expected and not missing and not extra else "failed"
    path_text = "" if path is None else f"; path={path}"
    return {
        "submodel_id": str(submodel_id),
        "check": str(check),
        "status": status,
        "message": (
            f"expected={sorted(expected)}; actual={sorted(actual)}; "
            f"missing={missing}; extra={extra}{path_text}"
        ),
    }


def handoff_ids_from_geojson(path: str | Path) -> set[str]:
    """Read SFINCS handoff IDs from a GeoJSON artifact, returning an empty set if absent."""
    path = Path(path)
    if not path.exists():
        return set()
    frame = gpd.read_file(path)
    if frame.empty or "sfincs_handoff_id" not in frame:
        return set()
    return set(frame["sfincs_handoff_id"].dropna().astype(str))


def handoff_artifact_report(
    submodels: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    sources,
    base_root: str | Path,
) -> pd.DataFrame:
    """Compare Wflow gauge/manifest handoff IDs against reviewed SFINCS source IDs."""
    base_root = Path(base_root)
    rows: list[dict[str, str]] = []
    for submodel in submodels:
        submodel_id = str(submodel.get("wflow_submodel_id", ""))
        manifest_ids = {str(value) for value in submodel.get("sfincs_handoff_ids", ()) if value}
        source_ids = handoff_source_ids_for_submodel(sources, submodel_id)
        gauges_path = base_root / submodel_id / "staticgeoms" / "gauges_sfincs.geojson"
        gauge_ids = handoff_ids_from_geojson(gauges_path)
        rows.append(
            handoff_artifact_row(
                submodel_id,
                "wflow_gauges_sfincs_match_sources",
                gauges_path,
                source_ids,
                gauge_ids,
            )
        )
        rows.append(
            handoff_artifact_row(
                submodel_id,
                "wflow_domain_manifest_matches_sources",
                None,
                source_ids,
                manifest_ids,
            )
        )
    return pd.DataFrame(rows)


def validate_wflow_sfincs_handoff_artifacts_current(
    config: dict,
    location_root,
    *,
    submodels: list[dict] | tuple[dict, ...] | None = None,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """Validate that Wflow gauges, SFINCS sources, and the manifest use one handoff set."""
    location_root = Path(location_root)
    if submodels is None:
        submodels = configured_or_manifest_submodels(config, location_root)
    base_root = resolve_location_path(
        location_root,
        config.get("wflow", {}).get("base_model_root", "data/wflow/base"),
    )
    sources = read_stream_boundary_handoff_location_artifacts(
        config,
        location_root,
        location_path=resolve_location_path,
    )
    report = handoff_artifact_report(submodels, sources, base_root)
    failed = report[report["status"].isin(["failed", "review_required"])] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.submodel_id}:{row.check}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Wflow-SFINCS handoff artifacts are stale or incomplete: {details}")
    return report


__all__ = [
    "handoff_artifact_report",
    "handoff_artifact_row",
    "handoff_ids_from_geojson",
    "handoff_source_ids_for_submodel",
    "validate_wflow_sfincs_handoff_artifacts_current",
]
