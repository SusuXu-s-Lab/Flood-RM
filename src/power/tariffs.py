"""Tariff selection for Marshfield REopt sizing.

The current Marshfield sandbox is in Eversource/NSTAR South Shore territory.
REopt should consume URDB labels, not project-local blended placeholders, when
we are making API-backed sizing runs.
"""

from __future__ import annotations

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
