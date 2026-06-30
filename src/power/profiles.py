"""Synthetic/OEDI-ready load-profile assignments.

The production artifact is ``load_profile_assignments.parquet``. Offline builds
use transparent synthetic 8760 overlays; live OEDI selection is deliberately an
adapter outside the core path.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import write_table

load_profile_assignments_schema_version = "stage_b_load_profile_assignments.v0.1"
hours_per_year = 8760
customer_class_critical = "critical_facility"
customer_class_residential = "residential"
customer_class_commercial = "commercial"
customer_class_industrial_proxy = "industrial_proxy"
default_ashrae_climate_zone = "ASHRAE_5A"

_facility_class_archetype: dict[str, tuple[str, str, str, str]] = {
    "police_eoc": ("comstock", "SmallOffice", "24x7", customer_class_critical),
    "fire_station": ("comstock", "SmallOffice", "24x7", customer_class_critical),
    "communications_exchange": ("comstock", "SmallOffice", "24x7", customer_class_critical),
    "responder_radio": ("comstock", "SmallOffice", "24x7", customer_class_critical),
    "municipal_facility": ("comstock", "SmallOffice", "business_hours", customer_class_critical),
    "senior_center": ("comstock", "MediumOffice", "business_hours", customer_class_critical),
    "school": ("comstock", "PrimarySchool", "school_calendar", customer_class_critical),
    "school_administration": ("comstock", "SmallOffice", "business_hours", customer_class_critical),
    "public_housing": ("resstock", "MultiFamily", "extended_residential", customer_class_residential),
    "healthcare_residential": ("comstock", "Outpatient", "extended_residential", customer_class_critical),
    "wastewater_treatment_plant": ("comstock", "Warehouse", "industrial_continuous", customer_class_critical),
    "public_works": ("comstock", "Warehouse", "business_hours", customer_class_critical),
}
_default_archetype = ("comstock", "SmallOffice", "business_hours", customer_class_critical)
synthetic_peak_kw_by_class = {
    "municipal_facility": 50.0,
    "police_eoc": 75.0,
    "fire_station": 60.0,
    "public_works": 80.0,
    "senior_center": 50.0,
    "school": 200.0,
    "public_housing": 120.0,
    "healthcare_residential": 200.0,
    "wastewater_treatment_plant": 500.0,
    "communications_exchange": 100.0,
    "responder_radio": 30.0,
}
tier_int_to_string = {0: "tier_0_life_safety", 1: "tier_1_response", 2: "tier_2_lifeline_support", 3: "tier_3_standard"}


@dataclass(frozen=True)
class LocationLoadProfileInputs:
    assignment_rows: list[dict[str, Any]]
    load_profiles_kw: dict[str, list[float]]
    load_profile_provenance_by_facility: dict[str, dict[str, Any]]
    electric_tariffs_by_facility: dict[str, dict[str, Any]]
    electric_tariff_provenance_by_facility: dict[str, dict[str, Any]]
    facility_lookup: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class UrdbTariffSelection:
    urdb_label: str
    utility: str
    rate_name: str
    sector: str
    service_type: str
    source_url: str
    effective_date: str
    applicability_status: str
    applicability_note: str

    def reopt_electric_tariff(self) -> dict[str, str]:
        return {"urdb_label": self.urdb_label}

    def provenance(self) -> dict[str, str]:
        return self.__dict__.copy()


eversource_south_shore_rate_source = "https://www.eversource.com/docs/default-source/rates-tariffs/ema-south-shore-rates.pdf"
south_shore_residential_r1_standard_offer = UrdbTariffSelection("698931e6d9dd764bf1013fef", "NSTAR Electric Company", "South Shore Residential R-1 Annual BS", "Residential", "Delivery with Standard Offer", eversource_south_shore_rate_source, "2026-02-01", "selected_by_customer_class", "Residential/public-housing fallback for resilience sizing.")
south_shore_general_g1_standard_offer = UrdbTariffSelection("698a2ef6918cc43ffc02be98", "NSTAR Electric Company", "South Shore General-G-1", "Industrial", "Delivery with Standard Offer", eversource_south_shore_rate_source, "2026-02-01", "needs_account_rate_confirmation", "Nonresidential critical-facility proxy until account rate is known.")
south_shore_medium_g2_standard_offer = UrdbTariffSelection("698a2c5679616684990babe8", "NSTAR Electric Company", "South Shore Medium General TOU G-2", "Commercial", "Delivery with Standard Offer", eversource_south_shore_rate_source, "2026-02-01", "selected_by_peak_kw", "Selected for 100 <= peak_kw < 500.")
south_shore_large_g3_standard_offer = UrdbTariffSelection("698a2df2fd2dc68d090eef78", "NSTAR Electric Company", "South Shore Large General TOU G-3", "Industrial", "Delivery with Standard Offer", eversource_south_shore_rate_source, "2026-02-01", "selected_by_peak_kw", "Selected for peak_kw >= 500.")


def assign_archetype(facility: Mapping[str, Any]) -> dict[str, Any]:
    source, btype, overlay, customer = _facility_class_archetype.get(str(facility.get("facility_class")), _default_archetype)
    return {"profile_source": source, "source_building_type": btype, "schedule_overlay": overlay, "customer_class": customer, "source_geography": default_ashrae_climate_zone, "facility_class": facility.get("facility_class")}


def build_load_profile_assignment_row(
    facility: Mapping[str, Any], *, load_asset_id: str, peak_kw: float, location_id: str,
    weather_year: int = 2018, rng_seed: int = 0, profile_source_version: str = "synthetic_overlay_v0",
    synthetic_placeholder: bool = True, profile_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    archetype = assign_archetype(facility)
    token = str(facility["facility_id"]).rsplit(":", 1)[-1]
    loadshape_id = f"{location_id}:loadshape:{token}:{archetype['profile_source']}:{archetype['source_building_type']}:{archetype['schedule_overlay']}"
    provenance = {"facility_id": facility["facility_id"], "facility_class": facility.get("facility_class"), "criticality_tier": facility.get("criticality_tier"), **archetype, "profile_source_version": profile_source_version, "peak_kw": peak_kw, "synthetic_placeholder": synthetic_placeholder}
    if profile_provenance:
        provenance["profile_provenance"] = dict(profile_provenance)
    return {
        "sandbox_id": location_id, "load_asset_id": load_asset_id, "loadshape_id": loadshape_id,
        "municipality_id": facility.get("municipality_id") or f"{location_id}:municipality:{location_id}",
        "tile_id": None, "feeder_id": facility.get("feeder_id"), "customer_class": archetype["customer_class"],
        "profile_source": archetype["profile_source"], "profile_source_version": profile_source_version,
        "source_geography": archetype["source_geography"], "source_building_type": archetype["source_building_type"],
        "weather_year": weather_year, "time_step_minutes": 60, "npts": hours_per_year,
        "p_scale_factor": 1.0, "q_scale_factor": 0.0, "annual_energy_kwh": None, "peak_kw": peak_kw,
        "power_factor_policy": "static_load_pf", "diversity_group_id": token, "rng_seed": rng_seed,
        "source_provenance": json.dumps(provenance, sort_keys=True), "schema_version": load_profile_assignments_schema_version,
    }


def build_archetype_load_profile(archetype: Mapping[str, Any], *, peak_kw: float, hours: int = hours_per_year) -> list[float]:
    overlay = archetype.get("schedule_overlay", "business_hours")
    fn = {"24x7": _shape_24x7, "business_hours": _shape_business_hours, "school_calendar": _shape_school_calendar, "extended_residential": _shape_extended_residential, "industrial_continuous": _shape_industrial_continuous}.get(str(overlay), _shape_business_hours)
    return [round(float(peak_kw) * fn(hour), 6) for hour in range(hours)]


def _shape_24x7(hour: int) -> float:
    return min(1.0, max(0.62, 0.8 + 0.2 * math.sin(math.pi * ((hour % 24) - 1) / 24)))


def _shape_business_hours(hour: int) -> float:
    dow, hod = (hour // 24) % 7, hour % 24
    value = 0.18 + 0.82 * math.sin(math.pi * ((hod - 9 + 0.5) / 9)) if 9 <= hod < 18 else 0.18
    if dow >= 5:
        value = min(value, 0.30)
    return 1.0 if hour == 61 else max(0.0, value)


def _shape_school_calendar(hour: int) -> float:
    dow, doy, hod = (hour // 24) % 7, (hour // 24) % 365, hour % 24
    if dow >= 5 or 165 <= doy < 245:
        return 0.05
    return 0.10 + 0.90 * math.sin(math.pi * ((hod - 7 + 0.5) / 9)) if 7 <= hod < 16 else 0.10


def _shape_extended_residential(hour: int) -> float:
    hod = hour % 24
    value = 0.30 + 0.40 * math.exp(-((hod - 7) ** 2) / 6.0) + 0.70 * math.exp(-((hod - 19) ** 2) / 8.0)
    if (hour // 24) % 7 >= 5:
        value *= 1.10
    return min(1.0, value)


def _shape_industrial_continuous(hour: int) -> float:
    return max(0.85, min(1.0, 0.92 + 0.06 * math.sin(2 * math.pi * ((hour % 24) - 6) / 24)))


def select_eversource_south_shore_tariff(facility: Mapping[str, Any], *, peak_kw: float, customer_class: str) -> UrdbTariffSelection:
    if customer_class == customer_class_residential or facility.get("facility_class") == "public_housing":
        return south_shore_residential_r1_standard_offer
    if peak_kw >= 500.0:
        return south_shore_large_g3_standard_offer
    if peak_kw >= 100.0:
        return south_shore_medium_g2_standard_offer
    return south_shore_general_g1_standard_offer


def load_profile_schema() -> Any:
    import pyarrow as pa
    return pa.schema([
        pa.field("sandbox_id", pa.string(), nullable=False), pa.field("load_asset_id", pa.string(), nullable=False),
        pa.field("loadshape_id", pa.string(), nullable=False), pa.field("municipality_id", pa.string()),
        pa.field("tile_id", pa.string()), pa.field("feeder_id", pa.string()), pa.field("customer_class", pa.string(), nullable=False),
        pa.field("profile_source", pa.string(), nullable=False), pa.field("profile_source_version", pa.string(), nullable=False),
        pa.field("source_geography", pa.string(), nullable=False), pa.field("source_building_type", pa.string(), nullable=False),
        pa.field("weather_year", pa.int16(), nullable=False), pa.field("time_step_minutes", pa.int16(), nullable=False),
        pa.field("npts", pa.int32(), nullable=False), pa.field("p_scale_factor", pa.float64(), nullable=False),
        pa.field("q_scale_factor", pa.float64(), nullable=False), pa.field("annual_energy_kwh", pa.float64()),
        pa.field("peak_kw", pa.float64(), nullable=False), pa.field("power_factor_policy", pa.string(), nullable=False),
        pa.field("diversity_group_id", pa.string(), nullable=False), pa.field("rng_seed", pa.int64(), nullable=False),
        pa.field("source_provenance", pa.string(), nullable=False), pa.field("schema_version", pa.string(), nullable=False),
    ])


def load_inputs(
    critical_facility_records: list[Mapping[str, Any]], load_match_by_facility: Mapping[str, Mapping[str, Any]], *,
    load_profile_assignments_path: Path, oedi_profile_cache_dir: Path | None = None, use_oedi_load_profiles: bool = False,
) -> LocationLoadProfileInputs:
    if use_oedi_load_profiles:
        raise NotImplementedError("live OEDI profile download is intentionally outside the core path; pass preselected profiles or use synthetic overlays")
    rows: list[dict[str, Any]] = []
    profiles: dict[str, list[float]] = {}
    profile_prov: dict[str, dict[str, Any]] = {}
    tariffs: dict[str, dict[str, Any]] = {}
    tariff_prov: dict[str, dict[str, Any]] = {}
    lookup: dict[str, dict[str, Any]] = {}
    for facility in critical_facility_records:
        normalized = dict(facility)
        normalized["criticality_tier"] = tier_int_to_string.get(normalized.get("criticality_tier"), normalized.get("criticality_tier"))
        fid = str(normalized["facility_id"])
        lookup[fid] = normalized
        match = load_match_by_facility.get(fid)
        if match is None:
            continue
        peak_kw = synthetic_peak_kw_by_class.get(str(normalized.get("facility_class") or "municipal_facility"), 50.0)
        archetype = assign_archetype(normalized)
        profile = build_archetype_load_profile(archetype, peak_kw=peak_kw)
        tariff = select_eversource_south_shore_tariff(normalized, peak_kw=peak_kw, customer_class=archetype["customer_class"])
        provenance = {**archetype, "synthetic_placeholder": True}
        row = build_load_profile_assignment_row(normalized, load_asset_id=str(match["load_asset_id"]), peak_kw=peak_kw, location_id=str(normalized.get("sandbox_id") or fid.split(":", 1)[0]), profile_provenance=provenance)
        row["annual_energy_kwh"] = round(sum(profile), 6)
        rows.append(row)
        profiles[fid] = profile
        profile_prov[fid] = provenance
        tariffs[fid] = tariff.reopt_electric_tariff()
        tariff_prov[fid] = tariff.provenance()
    write_table(load_profile_assignments_path, rows, schema=load_profile_schema())
    return LocationLoadProfileInputs(rows, profiles, profile_prov, tariffs, tariff_prov, lookup)


def classify_eversource_customer_class(*, peak_kw: float) -> str:
    if peak_kw < 10.0:
        return customer_class_residential
    if peak_kw < 200.0:
        return customer_class_commercial
    return customer_class_industrial_proxy


def assign_nodal_load_profiles(loads: Sequence[Mapping[str, Any]], *, profile_pool: Mapping[str, Sequence[Mapping[str, Any]]], root_seed: int, location_id: str, municipality_id: str | None = None, weather_year: int = 2018) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in loads:
        peak_kw, feeder_id, load_name = float(record["kw"]), str(record["feeder_id"]), str(record["load_name"])
        customer = classify_eversource_customer_class(peak_kw=peak_kw)
        candidates = profile_pool.get(customer)
        if not candidates:
            raise ValueError(f"profile_pool missing entries for customer_class={customer!r}")
        seed = int.from_bytes(hashlib.sha256(f"{root_seed}|{feeder_id}|{customer}|{load_name}".encode()).digest()[:8], "big")
        archetype = dict(candidates[seed % len(candidates)])
        row = build_load_profile_assignment_row({"facility_id": f"nodal:{load_name}", "municipality_id": municipality_id}, load_asset_id=f"{location_id}:asset:loads:{load_name}", peak_kw=peak_kw, location_id=location_id, weather_year=weather_year, rng_seed=seed, profile_source_version=archetype.get("profile_source_version", "synthetic_overlay_v0"))
        row.update({"feeder_id": feeder_id, "customer_class": customer, "profile_source": archetype["profile_source"], "source_building_type": archetype["source_building_type"], "source_geography": archetype.get("source_geography", default_ashrae_climate_zone), "p_scale_factor": peak_kw, "q_scale_factor": float(record.get("kvar", 0.0) or 0.0), "diversity_group_id": f"{feeder_id}:{customer}"})
        rows.append(row)
    return rows
