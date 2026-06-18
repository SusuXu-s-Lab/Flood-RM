"""Nodal demand assignment for non-critical loads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from power.load_profiles import LOAD_PROFILE_ASSIGNMENTS_SCHEMA_VERSION


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
