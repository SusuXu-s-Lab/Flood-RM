# Load profile assignments

"""Stage B load profile assignment: ComStock/ResStock archetypes per facility.

Methodology: each configured critical facility is mapped to a ComStock or
ResStock archetype keyed by the configured ASHRAE climate zone.
The archetype metadata feeds `load_profile_assignments.parquet` v0.1 and
selects the schedule overlay used to shape per-facility 8760 placeholder
profiles for REopt sizing. Real ComStock/ResStock CSV pulls are deferred to
a later sub-slice; this module assigns the archetype and emits the
provenance so the assignment artifact is auditable now.

NLR-Distribution-Suite handoff: the protocol-locked
`load_profile_assignments.parquet` is the canonical assignment record. When
the assignment feeds a gdm ``DistributionSystem`` (for example at the
OpenDSS export / PowerModelsONM settings slice), per-load time series are
attached via the infrasys ``SingleTimeSeries`` API on each
``DistributionLoad`` component, which is the NLR-native time-series
mechanism. Archetype mapping itself has no NLR-native equivalent because it
is a sandbox-specific domain decision.
"""


import json
import math
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any


LOAD_PROFILE_ASSIGNMENTS_SCHEMA_VERSION = "stage_b_load_profile_assignments.v0.1"
HOURS_PER_YEAR = 8760

# School-calendar approximation: classes in session Mon-Fri Sep-mid Jun.
# Day-of-year boundaries (0-indexed) for summer break.
_SUMMER_BREAK_START_DOY = 165  # ~Jun 14
_SUMMER_BREAK_END_DOY = 245    # ~Sep 2


DEFAULT_ASHRAE_CLIMATE_ZONE = "ASHRAE_5A"

CUSTOMER_CLASS_CRITICAL = "critical_facility"
CUSTOMER_CLASS_RESIDENTIAL = "residential"

# Facility class -> archetype assignment.
#
# Each value is a tuple of (profile_source, source_building_type,
# schedule_overlay, customer_class). The schedule_overlay is the high-level
# pattern used to shape the placeholder 8760 until real ComStock/ResStock
# pulls land. Allowed overlays: "24x7", "business_hours", "school_calendar",
# "extended_residential", "industrial_continuous".
_FACILITY_CLASS_ARCHETYPE: "dict[str, tuple[str, str, str, str]]" = {
    # 24x7 emergency / response sites.
    "police_eoc": ("comstock", "SmallOffice", "24x7", CUSTOMER_CLASS_CRITICAL),
    "fire_station": ("comstock", "SmallOffice", "24x7", CUSTOMER_CLASS_CRITICAL),
    "communications_exchange": ("comstock", "SmallOffice", "24x7", CUSTOMER_CLASS_CRITICAL),
    "responder_radio": ("comstock", "SmallOffice", "24x7", CUSTOMER_CLASS_CRITICAL),
    # Business-hours municipal facilities.
    "municipal_facility": ("comstock", "SmallOffice", "business_hours", CUSTOMER_CLASS_CRITICAL),
    "senior_center": ("comstock", "MediumOffice", "business_hours", CUSTOMER_CLASS_CRITICAL),
    "school_administration": ("comstock", "SmallOffice", "business_hours", CUSTOMER_CLASS_CRITICAL),
    "post_office": ("comstock", "SmallOffice", "business_hours", CUSTOMER_CLASS_CRITICAL),
    "animal_shelter": ("comstock", "RetailStandalone", "business_hours", CUSTOMER_CLASS_CRITICAL),
    "public_library_shelter": ("comstock", "SmallOffice", "business_hours", CUSTOMER_CLASS_CRITICAL),
    "harbor_master": ("comstock", "SmallOffice", "business_hours", CUSTOMER_CLASS_CRITICAL),
    "municipal_airport": ("comstock", "SmallOffice", "business_hours", CUSTOMER_CLASS_CRITICAL),
    "public_works": ("comstock", "Warehouse", "business_hours", CUSTOMER_CLASS_CRITICAL),
    # K-12 schools follow school-calendar schedule.
    "school": ("comstock", "PrimarySchool", "school_calendar", CUSTOMER_CLASS_CRITICAL),
    # Residential housing (multi-family).
    "public_housing": ("resstock", "MultiFamily", "extended_residential", CUSTOMER_CLASS_RESIDENTIAL),
    "healthcare_residential": (
        "comstock",
        "Outpatient",
        "extended_residential",
        CUSTOMER_CLASS_CRITICAL,
    ),
    # Industrial / process loads.
    "wastewater_treatment_plant": (
        "comstock",
        "Warehouse",
        "industrial_continuous",
        CUSTOMER_CLASS_CRITICAL,
    ),
}

_DEFAULT_ARCHETYPE = ("comstock", "SmallOffice", "business_hours", CUSTOMER_CLASS_CRITICAL)

SYNTHETIC_PEAK_KW_BY_CLASS = {
    "municipal_facility": 50.0,
    "police_eoc": 75.0,
    "fire_station": 60.0,
    "public_works": 80.0,
    "senior_center": 50.0,
    "school": 200.0,
    "school_administration": 60.0,
    "public_housing": 120.0,
    "healthcare_residential": 200.0,
    "wastewater_treatment_plant": 500.0,
    "communications_exchange": 100.0,
    "responder_radio": 30.0,
    "post_office": 35.0,
    "animal_shelter": 25.0,
    "public_library_shelter": 60.0,
    "harbor_master": 25.0,
    "municipal_airport": 50.0,
}

TIER_INT_TO_STRING = {
    0: "tier_0_life_safety",
    1: "tier_1_response",
    2: "tier_2_lifeline_support",
    3: "tier_3_standard",
}


@dataclass(frozen=True)
class LocationLoadProfileInputs:
    """Prepared load-profile, tariff, and provenance channels for DER sizing."""

    assignment_rows: list[dict[str, Any]]
    load_profiles_kw: dict[str, list[float]]
    load_profile_provenance_by_facility: dict[str, dict[str, Any]]
    electric_tariffs_by_facility: dict[str, dict[str, Any]]
    electric_tariff_provenance_by_facility: dict[str, dict[str, Any]]
    facility_lookup: dict[str, dict[str, Any]]


def assign_archetype(facility: Mapping[str, Any]) -> dict[str, Any]:
    """Return archetype assignment metadata for a critical-facility row.

    Falls back to ComStock SmallOffice / business hours when the facility's
    `facility_class` is not in the explicit map.
    """

    facility_class = facility.get("facility_class")
    source, building_type, schedule, customer_class = _FACILITY_CLASS_ARCHETYPE.get(
        facility_class, _DEFAULT_ARCHETYPE
    )
    return {
        "profile_source": source,
        "source_building_type": building_type,
        "schedule_overlay": schedule,
        "customer_class": customer_class,
        "source_geography": DEFAULT_ASHRAE_CLIMATE_ZONE,
        "facility_class": facility_class,
    }


def _load_profile_facility_token(facility_id: str) -> str:
    return facility_id.rsplit(":", 1)[-1]


def build_load_profile_assignment_row(
    facility: Mapping[str, Any],
    *,
    load_asset_id: str,
    peak_kw: float,
    sandbox_id: str,
    weather_year: int = 2018,
    rng_seed: int = 0,
    p_scale_factor: float = 1.0,
    q_scale_factor: float = 0.0,
    power_factor_policy: str = "static_load_pf",
    diversity_group_id: str | None = None,
    feeder_id: str | None = None,
    municipality_id: str | None = None,
    tile_id: str | None = None,
    profile_source_version: str = "2024_release_1",
    synthetic_placeholder: bool = True,
    profile_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit one `load_profile_assignments.parquet` v0.1 row for a facility.

    The archetype is chosen via :func:`assign_archetype` and stamped into the
    row's metadata + provenance JSON; the actual 8760 profile generation is
    performed by :func:`build_archetype_load_profile`, keyed off the same
    archetype tuple so the loadshape ID and provenance stay consistent.
    """

    archetype = assign_archetype(facility)
    facility_token = _load_profile_facility_token(facility["facility_id"])
    municipality_id = municipality_id or facility.get("municipality_id") or f"{sandbox_id}:municipality:{sandbox_id}"
    loadshape_id = (
        f"{sandbox_id}:loadshape:{facility_token}:"
        f"{archetype['profile_source']}:{archetype['source_building_type']}:"
        f"{archetype['schedule_overlay']}"
    )

    provenance = {
        "facility_id": facility["facility_id"],
        "facility_class": facility.get("facility_class"),
        "criticality_tier": facility.get("criticality_tier"),
        "profile_source": archetype["profile_source"],
        "source_building_type": archetype["source_building_type"],
        "source_geography": archetype["source_geography"],
        "schedule_overlay": archetype["schedule_overlay"],
        "profile_source_version": profile_source_version,
        "peak_kw": peak_kw,
        "synthetic_placeholder": synthetic_placeholder,
        "synthetic_placeholder_reason": (
            "real ResStock/ComStock 8760 download deferred to a later sub-slice; "
            "this row's 8760 series is generated from the archetype schedule_overlay"
        )
        if synthetic_placeholder
        else None,
    }
    if profile_provenance is not None:
        provenance["profile_provenance"] = dict(profile_provenance)

    return {
        "sandbox_id": sandbox_id,
        "load_asset_id": load_asset_id,
        "loadshape_id": loadshape_id,
        "municipality_id": municipality_id,
        "tile_id": tile_id,
        "feeder_id": feeder_id,
        "customer_class": archetype["customer_class"],
        "profile_source": archetype["profile_source"],
        "profile_source_version": profile_source_version,
        "source_geography": archetype["source_geography"],
        "source_building_type": archetype["source_building_type"],
        "weather_year": weather_year,
        "time_step_minutes": 60,
        "npts": HOURS_PER_YEAR,
        "p_scale_factor": p_scale_factor,
        "q_scale_factor": q_scale_factor,
        "annual_energy_kwh": None,
        "peak_kw": peak_kw,
        "power_factor_policy": power_factor_policy,
        "diversity_group_id": diversity_group_id or facility_token,
        "rng_seed": rng_seed,
        "source_provenance": json.dumps(provenance, sort_keys=True),
        "schema_version": LOAD_PROFILE_ASSIGNMENTS_SCHEMA_VERSION,
    }


def build_archetype_load_profile(
    archetype: Mapping[str, Any],
    *,
    peak_kw: float,
    hours: int = HOURS_PER_YEAR,
) -> list[float]:
    """Generate a placeholder 8760 profile shaped by ``archetype['schedule_overlay']``.

    Five overlays are recognized; an unknown overlay falls back to
    ``business_hours``. The profile is a synthetic stand-in for ResStock /
    ComStock data and is replaced once the EULP CSV pull lands.
    """

    overlay = archetype.get("schedule_overlay", "business_hours")
    shape_fn = _OVERLAY_SHAPE_FNS.get(overlay, _shape_business_hours)
    return [round(peak_kw * shape_fn(hour), 6) for hour in range(hours)]


def _shape_24x7(hour: int) -> float:
    """24/7 facility: high baseline with mild diurnal modulation."""

    hour_of_day = hour % 24
    # 0.8 baseline + 0.2 sinusoid peaking at 13:00 gives min 0.62, max 1.00.
    diurnal = math.sin(math.pi * (hour_of_day - 1) / 24)
    return min(1.0, max(0.62, 0.8 + 0.2 * diurnal))


def _shape_business_hours(hour: int) -> float:
    """Commercial facility: peak weekday 9-17, low overnight, weekend reduced."""

    day_of_week = (hour // 24) % 7
    hour_of_day = hour % 24
    weekend = day_of_week >= 5

    if 9 <= hour_of_day < 18:
        phase = (hour_of_day - 9 + 0.5) / 9
        shape = math.sin(math.pi * phase)
        value = 0.18 + 0.82 * shape
    else:
        value = 0.18

    if weekend:
        value = min(value, 0.30)

    # Force a strict peak so peak_kw is reachable on at least one hour.
    if hour == 13 and day_of_week == 2:  # mid-day Wednesday
        value = 1.0

    return value


def _shape_school_calendar(hour: int) -> float:
    """K-12 school: weekday 7-16 Sep-mid Jun; weekends and summer near-zero."""

    day_of_week = (hour // 24) % 7
    day_of_year = (hour // 24) % 365
    hour_of_day = hour % 24

    summer_break = _SUMMER_BREAK_START_DOY <= day_of_year < _SUMMER_BREAK_END_DOY
    weekend = day_of_week >= 5

    if weekend or summer_break:
        return 0.05  # vacant-school overnight loads only

    if 7 <= hour_of_day < 16:
        phase = (hour_of_day - 7 + 0.5) / 9
        return 0.10 + 0.90 * math.sin(math.pi * phase)
    return 0.10


def _shape_extended_residential(hour: int) -> float:
    """Residential multi-family: morning + evening peaks, weekend modestly higher."""

    day_of_week = (hour // 24) % 7
    hour_of_day = hour % 24
    weekend = day_of_week >= 5

    # Morning peak around 07:00, evening peak around 19:00.
    morning = math.exp(-((hour_of_day - 7) ** 2) / 6.0)
    evening = math.exp(-((hour_of_day - 19) ** 2) / 8.0)
    baseline = 0.30
    value = baseline + 0.40 * morning + 0.70 * evening

    if weekend:
        value *= 1.10

    return min(1.0, value)


def _shape_industrial_continuous(hour: int) -> float:
    """Process load (e.g. wastewater treatment): near-constant year-round."""

    # 0.92 baseline, +/- 0.06 mild diurnal swing.
    hour_of_day = hour % 24
    swing = 0.06 * math.sin(2 * math.pi * (hour_of_day - 6) / 24)
    return max(0.85, min(1.0, 0.92 + swing))


_OVERLAY_SHAPE_FNS = {
    "24x7": _shape_24x7,
    "business_hours": _shape_business_hours,
    "school_calendar": _shape_school_calendar,
    "extended_residential": _shape_extended_residential,
    "industrial_continuous": _shape_industrial_continuous,
}


def build_location_load_profile_inputs(
    critical_facility_records: list[Mapping[str, Any]],
    critical_load_assignment_by_facility: Mapping[str, Mapping[str, Any]],
    *,
    load_profile_assignments_path: Path,
    oedi_profile_cache_dir: Path,
    use_oedi_load_profiles: bool = False,
) -> LocationLoadProfileInputs:
    """Build REopt-ready load profiles, assignment rows, and tariff metadata.

    The public interface takes artifact rows and writes the canonical
    `load_profile_assignments.parquet` table. Live OEDI/ResStock/ComStock pulls
    remain opt-in so offline notebook execution preserves the synthetic overlay.
    """

    import pyarrow as pa
    import pyarrow.parquet as pq


    assignment_rows: list[dict[str, Any]] = []
    load_profiles_kw: dict[str, list[float]] = {}
    load_profile_provenance_by_facility: dict[str, dict[str, Any]] = {}
    electric_tariffs_by_facility: dict[str, dict[str, Any]] = {}
    electric_tariff_provenance_by_facility: dict[str, dict[str, Any]] = {}
    facility_lookup: dict[str, dict[str, Any]] = {}

    for facility in critical_facility_records:
        normalized = dict(facility)
        raw_tier = normalized.get("criticality_tier")
        normalized["criticality_tier"] = TIER_INT_TO_STRING.get(raw_tier, raw_tier)
        facility_id = str(normalized["facility_id"])
        facility_lookup[facility_id] = normalized
        critical_assignment = critical_load_assignment_by_facility.get(facility_id)
        if critical_assignment is None:
            continue

        facility_class = normalized.get("facility_class") or "municipal_facility"
        peak_kw = SYNTHETIC_PEAK_KW_BY_CLASS.get(str(facility_class), 50.0)
        archetype = assign_archetype(normalized)
        tariff_selection = select_eversource_south_shore_tariff(
            normalized,
            peak_kw=peak_kw,
            customer_class=archetype["customer_class"],
        )
        electric_tariffs_by_facility[facility_id] = tariff_selection.reopt_electric_tariff()
        electric_tariff_provenance_by_facility[facility_id] = tariff_selection.provenance()

        if use_oedi_load_profiles:
            profile_selection = select_oedi_profile(
                archetype,
                cache_dir=oedi_profile_cache_dir,
                selection_token=facility_id,
            )
            profile = load_oedi_8760_profile_kw(
                profile_selection,
                cache_dir=oedi_profile_cache_dir,
                target_peak_kw=peak_kw,
            )
            profile_source_version = profile_selection.release
            load_profile_provenance_by_facility[facility_id] = {
                **profile_selection.provenance(),
                "synthetic_placeholder": False,
            }
        else:
            profile = build_archetype_load_profile(archetype, peak_kw=peak_kw)
            profile_source_version = "synthetic_overlay_v0"
            load_profile_provenance_by_facility[facility_id] = {
                "profile_source": archetype["profile_source"],
                "source_building_type": archetype["source_building_type"],
                "source_geography": archetype["source_geography"],
                "schedule_overlay": archetype["schedule_overlay"],
                "synthetic_placeholder": True,
            }

        row = build_load_profile_assignment_row(
            normalized,
            load_asset_id=str(critical_assignment["load_asset_id"]),
            peak_kw=peak_kw,
            sandbox_id=str(normalized.get("sandbox_id") or facility_id.split(":", 1)[0]),
            profile_source_version=profile_source_version,
            synthetic_placeholder=not use_oedi_load_profiles,
            profile_provenance=load_profile_provenance_by_facility[facility_id],
        )
        row["annual_energy_kwh"] = round(sum(profile), 6)
        assignment_rows.append(row)
        load_profiles_kw[facility_id] = profile

    schema = load_profile_assignments_pyarrow_schema()
    columns = {field.name: [row.get(field.name) for row in assignment_rows] for field in schema}
    load_profile_assignments_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=schema), load_profile_assignments_path)

    return LocationLoadProfileInputs(
        assignment_rows=assignment_rows,
        load_profiles_kw=load_profiles_kw,
        load_profile_provenance_by_facility=load_profile_provenance_by_facility,
        electric_tariffs_by_facility=electric_tariffs_by_facility,
        electric_tariff_provenance_by_facility=electric_tariff_provenance_by_facility,
        facility_lookup=facility_lookup,
    )


def load_profile_assignments_pyarrow_schema() -> Any:
    """Return the pyarrow schema for `load_profile_assignments.parquet` v0.1.

    Field types and nullability mirror `simulated_data_protocol.md` so rows
    produced by :func:`build_load_profile_assignment_row` serialise without
    coercion. `annual_energy_kwh` is nullable until the actual 8760 series is
    integrated to produce the value.
    """

    import pyarrow as pa

    return pa.schema(
        [
            pa.field("sandbox_id", pa.string(), nullable=False),
            pa.field("load_asset_id", pa.string(), nullable=False),
            pa.field("loadshape_id", pa.string(), nullable=False),
            pa.field("municipality_id", pa.string(), nullable=True),
            pa.field("tile_id", pa.string(), nullable=True),
            pa.field("feeder_id", pa.string(), nullable=True),
            pa.field("customer_class", pa.string(), nullable=False),
            pa.field("profile_source", pa.string(), nullable=False),
            pa.field("profile_source_version", pa.string(), nullable=False),
            pa.field("source_geography", pa.string(), nullable=False),
            pa.field("source_building_type", pa.string(), nullable=False),
            pa.field("weather_year", pa.int16(), nullable=False),
            pa.field("time_step_minutes", pa.int16(), nullable=False),
            pa.field("npts", pa.int32(), nullable=False),
            pa.field("p_scale_factor", pa.float64(), nullable=False),
            pa.field("q_scale_factor", pa.float64(), nullable=False),
            pa.field("annual_energy_kwh", pa.float64(), nullable=True),
            pa.field("peak_kw", pa.float64(), nullable=False),
            pa.field("power_factor_policy", pa.string(), nullable=False),
            pa.field("diversity_group_id", pa.string(), nullable=False),
            pa.field("rng_seed", pa.int64(), nullable=False),
            pa.field("source_provenance", pa.string(), nullable=False),
            pa.field("schema_version", pa.string(), nullable=False),
        ]
    )


# Nodal load profile support

"""Nodal demand assignment for non-critical loads."""


import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any



EVERSOURCE_MA_RESIDENTIAL_PEAK_KW_CEILING = 10.0
EVERSOURCE_MA_INDUSTRIAL_PEAK_KW_FLOOR = 200.0

CUSTOMER_CLASS_RESIDENTIAL = "residential"
CUSTOMER_CLASS_COMMERCIAL = "commercial"
CUSTOMER_CLASS_INDUSTRIAL_PROXY = "industrial_proxy"

_HOURS_PER_YEAR = 8760
_DEFAULT_WEATHER_YEAR = 2018
_DEFAULT_POWER_FACTOR_POLICY = "static_load_pf"


def classify_eversource_customer_class(*, peak_kw: float) -> str:
    """Classify a load by peak kW using Eversource MA rate-class thresholds.

    Returns one of ``residential``, ``commercial``, or ``industrial_proxy``.
    """

    if peak_kw < EVERSOURCE_MA_RESIDENTIAL_PEAK_KW_CEILING:
        return CUSTOMER_CLASS_RESIDENTIAL
    if peak_kw < EVERSOURCE_MA_INDUSTRIAL_PEAK_KW_FLOOR:
        return CUSTOMER_CLASS_COMMERCIAL
    return CUSTOMER_CLASS_INDUSTRIAL_PROXY


def assign_nodal_load_profiles(
    loads: Sequence[Mapping[str, Any]],
    *,
    profile_pool: Mapping[str, Sequence[Mapping[str, Any]]],
    root_seed: int,
    sandbox_id: str,
    municipality_id: str | None = None,
    weather_year: int = _DEFAULT_WEATHER_YEAR,
    power_factor_policy: str = _DEFAULT_POWER_FACTOR_POLICY,
) -> list[dict[str, Any]]:
    """Assign one ``load_profile_assignments.parquet`` row per nodal load.

    Each load record must carry ``load_name``, ``bus``, ``kw``, ``kvar``, and
    ``feeder_id``. ``profile_pool`` maps each customer class to a non-empty
    list of archetype dicts (``profile_source``, ``source_building_type``,
    ``schedule_overlay``, ``source_geography``, ``profile_source_version``).

    The returned rows are ordered to match ``loads``.
    """

    municipality_id = municipality_id or f"{sandbox_id}:municipality:{sandbox_id}"
    rows: list[dict[str, Any]] = []
    for record in loads:
        peak_kw = float(record["kw"])
        kvar = float(record.get("kvar", 0.0))
        feeder_id = str(record["feeder_id"])
        load_name = str(record["load_name"])
        customer_class = classify_eversource_customer_class(peak_kw=peak_kw)

        candidates = profile_pool.get(customer_class)
        if not candidates:
            raise ValueError(
                f"profile_pool missing entries for customer_class={customer_class!r}"
            )

        diversity_group_id = f"{feeder_id}:{customer_class}"
        selection_seed = _deterministic_selection_seed(
            root_seed=root_seed,
            feeder_id=feeder_id,
            customer_class=customer_class,
            load_name=load_name,
        )
        archetype = candidates[selection_seed % len(candidates)]

        loadshape_id = (
            f"{sandbox_id}:loadshape:nodal:{feeder_id}:{customer_class}:"
            f"{archetype['profile_source']}:{archetype['source_building_type']}:{load_name}"
        )

        provenance = {
            "classifier": "eversource_ma_rate_class_kw_threshold",
            "diversity_rule": "deterministic_hash_within_feeder_class_bucket",
            "profile_source": archetype["profile_source"],
            "source_building_type": archetype["source_building_type"],
            "schedule_overlay": archetype.get("schedule_overlay"),
            "source_geography": archetype.get("source_geography", "ASHRAE_5A"),
            "profile_source_version": archetype.get(
                "profile_source_version", "synthetic_overlay_v0"
            ),
            "selection_seed": selection_seed,
            "synthetic_placeholder": True,
        }

        rows.append(
            {
                "sandbox_id": sandbox_id,
                "load_asset_id": f"{sandbox_id}:asset:loads:{load_name}",
                "loadshape_id": loadshape_id,
                "municipality_id": municipality_id,
                "tile_id": None,
                "feeder_id": feeder_id,
                "customer_class": customer_class,
                "profile_source": archetype["profile_source"],
                "profile_source_version": archetype.get(
                    "profile_source_version", "synthetic_overlay_v0"
                ),
                "source_geography": archetype.get("source_geography", "ASHRAE_5A"),
                "source_building_type": archetype["source_building_type"],
                "weather_year": weather_year,
                "time_step_minutes": 60,
                "npts": _HOURS_PER_YEAR,
                "p_scale_factor": peak_kw,
                "q_scale_factor": kvar,
                "annual_energy_kwh": None,
                "peak_kw": peak_kw,
                "power_factor_policy": power_factor_policy,
                "diversity_group_id": diversity_group_id,
                "rng_seed": selection_seed,
                "source_provenance": json.dumps(provenance, sort_keys=True),
                "schema_version": LOAD_PROFILE_ASSIGNMENTS_SCHEMA_VERSION,
            }
        )

    return rows


def _deterministic_selection_seed(
    *, root_seed: int, feeder_id: str, customer_class: str, load_name: str
) -> int:
    token = f"{root_seed}|{feeder_id}|{customer_class}|{load_name}"
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


# Tariff selection

"""Tariff selection for Marshfield REopt sizing.

The current Marshfield sandbox is in Eversource/NSTAR South Shore territory.
REopt should consume URDB labels, not project-local blended placeholders, when
we are making API-backed sizing runs.
"""


from dataclasses import dataclass
from typing import Any


EVERSOURCE_SOUTH_SHORE_RATE_SOURCE = (
    "https://www.eversource.com/docs/default-source/rates-tariffs/"
    "ema-south-shore-rates.pdf?sfvrsn=f6b2b9ce_16"
)


@dataclass(frozen=True)
class UrdbTariffSelection:
    """Selected URDB tariff plus applicability metadata for provenance."""

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
        return {
            "urdb_label": self.urdb_label,
            "utility": self.utility,
            "rate_name": self.rate_name,
            "sector": self.sector,
            "service_type": self.service_type,
            "source_url": self.source_url,
            "effective_date": self.effective_date,
            "applicability_status": self.applicability_status,
            "applicability_note": self.applicability_note,
        }


SOUTH_SHORE_RESIDENTIAL_R1_STANDARD_OFFER = UrdbTariffSelection(
    urdb_label="698931e6d9dd764bf1013fef",
    utility="NSTAR Electric Company",
    rate_name="South Shore Residential R-1 Annual BS (32)",
    sector="Residential",
    service_type="Delivery with Standard Offer",
    source_url=EVERSOURCE_SOUTH_SHORE_RATE_SOURCE,
    effective_date="2026-02-01",
    applicability_status="selected_by_customer_class",
    applicability_note="Residential/public-housing fallback for resilience sizing.",
)

SOUTH_SHORE_GENERAL_G1_STANDARD_OFFER = UrdbTariffSelection(
    urdb_label="698a2ef6918cc43ffc02be98",
    utility="NSTAR Electric Company",
    rate_name="South Shore General-G-1 (33)",
    sector="Industrial",
    service_type="Delivery with Standard Offer",
    source_url=EVERSOURCE_SOUTH_SHORE_RATE_SOURCE,
    effective_date="2026-02-01",
    applicability_status="needs_account_rate_confirmation",
    applicability_note=(
        "Used for nonresidential critical facilities below 100 kW until an "
        "actual account tariff is known; URDB metadata lists 100 kW minimum."
    ),
)

SOUTH_SHORE_MEDIUM_G2_STANDARD_OFFER = UrdbTariffSelection(
    urdb_label="698a2c5679616684990babe8",
    utility="NSTAR Electric Company",
    rate_name="South Shore Medium General Time-of-Use G-2 BS (84)",
    sector="Commercial",
    service_type="Delivery with Standard Offer",
    source_url=EVERSOURCE_SOUTH_SHORE_RATE_SOURCE,
    effective_date="2026-02-01",
    applicability_status="selected_by_peak_kw",
    applicability_note="Selected for nonresidential critical facilities with 100 <= peak_kw < 500.",
)

SOUTH_SHORE_LARGE_G3_STANDARD_OFFER = UrdbTariffSelection(
    urdb_label="698a2df2fd2dc68d090eef78",
    utility="NSTAR Electric Company",
    rate_name="South Shore Large General Time-of-Use G-3 BS (24)",
    sector="Industrial",
    service_type="Delivery with Standard Offer",
    source_url=EVERSOURCE_SOUTH_SHORE_RATE_SOURCE,
    effective_date="2026-02-01",
    applicability_status="selected_by_peak_kw",
    applicability_note="Selected for nonresidential critical facilities with peak_kw >= 500.",
)


def select_eversource_south_shore_tariff(
    facility: dict[str, Any],
    *,
    peak_kw: float,
    customer_class: str,
) -> UrdbTariffSelection:
    """Choose a documented URDB tariff label for Marshfield REopt sizing.

    This is still a modelling assumption: the proper final assignment is each
    facility's actual utility account rate. Until that exists, use current
    South Shore standard-offer labels and carry applicability status in
    provenance so cost results are not overclaimed.
    """

    if customer_class == "residential" or facility.get("facility_class") == "public_housing":
        return SOUTH_SHORE_RESIDENTIAL_R1_STANDARD_OFFER
    if peak_kw >= 500.0:
        return SOUTH_SHORE_LARGE_G3_STANDARD_OFFER
    if peak_kw >= 100.0:
        return SOUTH_SHORE_MEDIUM_G2_STANDARD_OFFER
    return SOUTH_SHORE_GENERAL_G1_STANDARD_OFFER


# OEDI stock profile support

"""OEDI ResStock/ComStock profile access for REopt load inputs."""


import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


OEDI_BUCKET_HTTPS_ROOT = "https://oedi-data-lake.s3.amazonaws.com"
EULP_DATASET_PREFIX = (
    "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021"
)
EULP_RELEASE_BY_SOURCE = {
    "comstock": "comstock_amy2018_release_1",
    "resstock": "resstock_amy2018_release_1",
}
PLYMOUTH_COUNTY_GEOID = "G2500230"
EULP_TOTAL_ELECTRICITY_COLUMN = "out.electricity.total.energy_consumption"


@dataclass(frozen=True)
class OediProfileSelection:
    """A selected public EULP building profile."""

    profile_source: str
    release: str
    county_id: str
    building_id: str
    source_building_type: str
    metadata_filter_column: str | None
    profile_url: str
    metadata_url: str
    selection_method: str

    def provenance(self) -> dict[str, str | None]:
        return {
            "profile_source": self.profile_source,
            "profile_source_version": self.release,
            "county_id": self.county_id,
            "building_id": self.building_id,
            "source_building_type": self.source_building_type,
            "metadata_filter_column": self.metadata_filter_column,
            "profile_url": self.profile_url,
            "metadata_url": self.metadata_url,
            "selection_method": self.selection_method,
        }


def metadata_url(profile_source: str) -> str:
    release = EULP_RELEASE_BY_SOURCE[profile_source]
    return f"{OEDI_BUCKET_HTTPS_ROOT}/{EULP_DATASET_PREFIX}/{release}/metadata/metadata.parquet"


def profile_url(profile_source: str, *, county_id: str, building_id: str) -> str:
    release = EULP_RELEASE_BY_SOURCE[profile_source]
    return (
        f"{OEDI_BUCKET_HTTPS_ROOT}/{EULP_DATASET_PREFIX}/{release}/"
        "timeseries_individual_buildings/by_county/upgrade=0/"
        f"county={county_id}/{building_id}-0.parquet"
    )


def download_if_missing(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            path.write_bytes(response.read())
    return path


def load_oedi_metadata(profile_source: str, *, cache_dir: Path) -> pd.DataFrame:
    url = metadata_url(profile_source)
    path = cache_dir / "metadata" / profile_source / "metadata.parquet"
    download_if_missing(url, path)
    return pd.read_parquet(path)


def select_oedi_profile(
    archetype: dict[str, Any],
    *,
    cache_dir: Path,
    county_id: str = PLYMOUTH_COUNTY_GEOID,
    selection_token: str,
) -> OediProfileSelection:
    """Select a deterministic EULP individual-building profile."""

    profile_source = str(archetype["profile_source"])
    release = EULP_RELEASE_BY_SOURCE[profile_source]
    source_building_type = str(archetype["source_building_type"])
    metadata = load_oedi_metadata(profile_source, cache_dir=cache_dir)

    county_filtered = _filter_county(metadata, county_id)
    type_column = _building_type_column(county_filtered)
    if type_column is not None:
        type_filtered = county_filtered[county_filtered[type_column] == source_building_type]
    else:
        type_filtered = county_filtered
    candidates = type_filtered if not type_filtered.empty else county_filtered
    if candidates.empty:
        raise ValueError(f"No {profile_source} EULP metadata rows found for county {county_id}")

    building_id = _stable_building_id(candidates.index, selection_token)
    return OediProfileSelection(
        profile_source=profile_source,
        release=release,
        county_id=county_id,
        building_id=building_id,
        source_building_type=source_building_type,
        metadata_filter_column=type_column,
        profile_url=profile_url(profile_source, county_id=county_id, building_id=building_id),
        metadata_url=metadata_url(profile_source),
        selection_method="county_building_type_stable_hash",
    )


def load_oedi_8760_profile_kw(
    selection: OediProfileSelection,
    *,
    cache_dir: Path,
    target_peak_kw: float | None = None,
) -> list[float]:
    """Load a selected 15-minute EULP profile and convert it to hourly kW."""

    path = cache_dir / "profiles" / selection.profile_source / selection.county_id / (
        f"{selection.building_id}-0.parquet"
    )
    download_if_missing(selection.profile_url, path)
    frame = pd.read_parquet(path, columns=[EULP_TOTAL_ELECTRICITY_COLUMN])
    values = frame[EULP_TOTAL_ELECTRICITY_COLUMN].astype(float).to_list()
    if len(values) % 4 != 0:
        raise ValueError(f"Expected 15-minute EULP profile length divisible by 4; got {len(values)}")
    hourly_kw = [sum(values[idx : idx + 4]) for idx in range(0, len(values), 4)]
    if len(hourly_kw) != 8760:
        raise ValueError(f"Expected 8760 hourly values after aggregation; got {len(hourly_kw)}")
    if target_peak_kw is not None and max(hourly_kw) > 0:
        scale = target_peak_kw / max(hourly_kw)
        hourly_kw = [value * scale for value in hourly_kw]
    return [round(value, 6) for value in hourly_kw]


def _filter_county(metadata: pd.DataFrame, county_id: str) -> pd.DataFrame:
    if "in.county" in metadata.columns:
        filtered = metadata[metadata["in.county"] == county_id]
        if not filtered.empty:
            return filtered
    if "in.resstock_county_id" in metadata.columns:
        # SHIFT's example uses the human-readable county field for first pass.
        filtered = metadata[metadata["in.resstock_county_id"] == "MA, Plymouth County"]
        if not filtered.empty:
            return filtered
    return metadata


def _building_type_column(metadata: pd.DataFrame) -> str | None:
    for column in ("in.building_type", "in.geometry_building_type_recs"):
        if column in metadata.columns:
            return column
    return None


def _stable_building_id(index: pd.Index, token: str) -> str:
    labels = [str(value) for value in index]
    labels.sort()
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return labels[int(digest[:12], 16) % len(labels)]


load_inputs = build_location_load_profile_inputs
