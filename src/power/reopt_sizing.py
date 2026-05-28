"""Artifact-level Layer 2 REopt sizing for the DER inventory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from power.der_inventory import (
    DEFAULT_TIER_CLF_MAP,
    FEMA_COMMUNITY_LIFELINES_OUTAGE_HOURS,
    apply_layer_2_reopt_sizing,
    der_inventory_pyarrow_schema,
)
from power.load_profiles import TIER_INT_TO_STRING
from power.load_profiles import build_archetype_load_profile
from power.tariffs import select_eversource_south_shore_tariff


@dataclass(frozen=True)
class ReoptSizingResult:
    """Summary of an artifact-level REopt sizing refresh."""

    der_inventory_path: Path
    total_rows: int
    attempted_rows: int
    reopt_sized_rows: int
    provisional_rows: int


class CachedReoptResultsClient:
    """Cache-only REopt client for replaying committed local REopt artifacts."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        if not path.exists():
            raise FileNotFoundError(f"REopt cache miss for payload digest {digest}: {path}")
        return json.loads(path.read_text(encoding="utf-8"))


def run_layer_2_reopt_sizing(
    *,
    smart_ds_compat_dir: Path,
    reopt_client: Any,
    outage_duration_hours: int = FEMA_COMMUNITY_LIFELINES_OUTAGE_HOURS,
    outage_start_hour: int = 4392,
    live_limit: int | None = None,
) -> ReoptSizingResult:
    """Apply REopt Layer 2 sizing to `der_inventory.parquet` artifacts."""

    der_inventory_path = smart_ds_compat_dir / "der_inventory.parquet"
    der_rows = pd.read_parquet(der_inventory_path).to_dict(orient="records")
    if live_limit is None:
        rows_for_reopt = der_rows
    else:
        rows_for_reopt = der_rows[:live_limit]

    inputs = _load_reopt_inputs(smart_ds_compat_dir)
    sized_subset = apply_layer_2_reopt_sizing(
        rows_for_reopt,
        facility_lookup=inputs["facility_lookup"],
        load_profiles_kw=inputs["load_profiles_kw"],
        reopt_client=reopt_client,
        electric_tariffs=inputs["electric_tariffs_by_facility"],
        electric_tariff_provenance=inputs["electric_tariff_provenance_by_facility"],
        load_profile_provenance=inputs["load_profile_provenance_by_facility"],
        tier_clf_map=DEFAULT_TIER_CLF_MAP,
        outage_duration_hours=outage_duration_hours,
        outage_start_hour=outage_start_hour,
    )
    sized_by_der_id = {row["der_id"]: row for row in sized_subset}
    sized_rows = [sized_by_der_id.get(row["der_id"], dict(row)) for row in der_rows]

    _write_der_inventory(der_inventory_path, sized_rows)
    reopt_sized_rows = sum(
        1 for row in sized_rows if row.get("placement_rule") == "reopt_resilience_sizing"
    )
    return ReoptSizingResult(
        der_inventory_path=der_inventory_path,
        total_rows=len(sized_rows),
        attempted_rows=len(rows_for_reopt),
        reopt_sized_rows=reopt_sized_rows,
        provisional_rows=len(sized_rows) - reopt_sized_rows,
    )


def _load_reopt_inputs(smart_ds_compat_dir: Path) -> dict[str, Any]:
    facilities = pd.read_parquet(smart_ds_compat_dir / "critical_facilities.parquet")
    assignments = pd.read_parquet(smart_ds_compat_dir / "load_profile_assignments.parquet")

    facility_lookup: dict[str, dict[str, Any]] = {}
    for row in facilities.to_dict(orient="records"):
        normalized = dict(row)
        raw_tier = normalized.get("criticality_tier")
        normalized["criticality_tier"] = TIER_INT_TO_STRING.get(raw_tier, raw_tier)
        facility_lookup[str(normalized["facility_id"])] = normalized

    load_profiles_kw: dict[str, list[float]] = {}
    load_profile_provenance_by_facility: dict[str, dict[str, Any]] = {}
    electric_tariffs_by_facility: dict[str, dict[str, Any]] = {}
    electric_tariff_provenance_by_facility: dict[str, dict[str, Any]] = {}

    for row in assignments.to_dict(orient="records"):
        provenance = json.loads(row["source_provenance"])
        facility_id = str(provenance["facility_id"])
        archetype = {
            "profile_source": row["profile_source"],
            "source_building_type": row["source_building_type"],
            "source_geography": row["source_geography"],
            "schedule_overlay": provenance.get("schedule_overlay", "business_hours"),
        }
        peak_kw = float(row["peak_kw"])
        load_profiles_kw[facility_id] = build_archetype_load_profile(archetype, peak_kw=peak_kw)
        load_profile_provenance_by_facility[facility_id] = {
            "profile_source": row["profile_source"],
            "profile_source_version": row.get("profile_source_version"),
            "source_building_type": row["source_building_type"],
            "source_geography": row["source_geography"],
            "schedule_overlay": archetype["schedule_overlay"],
            "loadshape_id": row.get("loadshape_id"),
            "synthetic_placeholder": bool(provenance.get("synthetic_placeholder", False)),
        }
        facility = facility_lookup[facility_id]
        tariff_selection = select_eversource_south_shore_tariff(
            facility,
            peak_kw=peak_kw,
            customer_class=str(row["customer_class"]),
        )
        electric_tariffs_by_facility[facility_id] = tariff_selection.reopt_electric_tariff()
        electric_tariff_provenance_by_facility[facility_id] = tariff_selection.provenance()

    return {
        "facility_lookup": facility_lookup,
        "load_profiles_kw": load_profiles_kw,
        "load_profile_provenance_by_facility": load_profile_provenance_by_facility,
        "electric_tariffs_by_facility": electric_tariffs_by_facility,
        "electric_tariff_provenance_by_facility": electric_tariff_provenance_by_facility,
    }


def _write_der_inventory(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = der_inventory_pyarrow_schema()
    columns = {
        field.name: [_clean_missing(row.get(field.name)) for row in rows]
        for field in schema
    }
    pq.write_table(pa.table(columns, schema=schema), path)


def _clean_missing(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
