"""Nodal demand assignment for non-critical loads.

Closes the gap between synthetic grid loads and load-profile assignments.
Critical-facility load assignment lives in
``dft.power.load_profiles``; this module handles every other load.

Two design rules are taken from the protocol and cited literature:

1. **Customer-class classification by Eversource MA rate-class kW thresholds.**
   Marshfield is served by Eversource Massachusetts (NSTAR Electric). Their
   published retail tariff (MDPU 1310) uses these breakpoints, which match the
   tariff selector already in ``dft.power.tariffs``:

   * Rates R-1 / R-2 (residential): typical residential customers, single-phase
     service with peak demand below ~10 kW.
   * Rate G-1 (General Service Small): commercial customers with monthly
     demand under 10 kW.
   * Rate G-2 (General Service Medium): commercial customers with demand
     between 10 kW and 200 kW.
   * Rate G-3 (General Service Large) / industrial: customers with demand at
     or above 200 kW.

   We collapse R-* + G-1 below 10 kW into ``residential``, G-2 into
   ``commercial``, and G-3 / industrial-tariff customers into
   ``industrial_proxy`` so that the customer class lines up with the
   ResStock / ComStock archetype families already documented in
   ``simulated_data_protocol.md`` §"Stage B Load Profile Method".

2. **Diversity-preserving deterministic sampling.** Within each
   ``(feeder_id, customer_class)`` bucket each load draws an archetype from
   the supplied profile pool by hashing ``(root_seed, feeder_id,
   customer_class, load_name)``. Zhu & Mather (R12) show that assigning a
   single scaled feeder curve to every nodal load destroys the diversity and
   variability QSTS and MPC studies depend on; this implementation enforces
   that anti-uniform-scaling rule deterministically and reproducibly.

The emitted rows follow the ``load_profile_assignments.parquet`` v0.1 schema
defined in ``dft.power.load_profiles`` so nodal rows concatenate cleanly with
the critical-facility rows produced by
``build_location_load_profile_inputs``.

Citations:

* Eversource Massachusetts retail electric rates, MDPU 1310 tariff schedules
  (R-1, R-2, G-1, G-2, G-3): https://www.eversource.com/content/residential/account-billing/manage-bill/about-your-bill/rates-tariffs/massachusetts-electric
* Zhu, X. and Mather, B. "Data-Driven Load Diversity and Variability Modeling
  for Quasi-Static Time-Series Simulation on Distribution Feeders." NREL.
* Wilson, E. et al. "End-Use Load Profiles for the U.S. Building Stock."
  NREL/TP-5500-80889, 2022.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from power.load_profiles import LOAD_PROFILE_ASSIGNMENTS_SCHEMA_VERSION


EVERSOURCE_MA_RESIDENTIAL_PEAK_KW_CEILING = 10.0
EVERSOURCE_MA_INDUSTRIAL_PEAK_KW_FLOOR = 200.0

EVERSOURCE_CLASSIFIER_CITATION = (
    "Eversource Massachusetts retail electric tariff schedules MDPU 1310: "
    "R-1/R-2 residential service (typical peak <10 kW), G-1 general service "
    "small (<10 kW), G-2 medium (10-200 kW), G-3 large/industrial (>=200 kW)."
)

CUSTOMER_CLASS_RESIDENTIAL = "residential"
CUSTOMER_CLASS_COMMERCIAL = "commercial"
CUSTOMER_CLASS_INDUSTRIAL_PROXY = "industrial_proxy"

_HOURS_PER_YEAR = 8760
_DEFAULT_WEATHER_YEAR = 2018
_DEFAULT_POWER_FACTOR_POLICY = "static_load_pf"


def classify_eversource_customer_class(*, peak_kw: float) -> str:
    """Classify a load by peak kW using Eversource MA rate-class thresholds.

    See module docstring for citation. Returns one of ``residential``,
    ``commercial``, or ``industrial_proxy``.
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
            "classifier_citation": EVERSOURCE_CLASSIFIER_CITATION,
            "diversity_rule": "deterministic_hash_within_feeder_class_bucket",
            "diversity_rule_citation": (
                "Zhu & Mather, 'Data-Driven Load Diversity and Variability "
                "Modeling for Quasi-Static Time-Series Simulation on "
                "Distribution Feeders' (NREL)."
            ),
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
