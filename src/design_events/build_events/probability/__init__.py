"""Probability-model seam for Event Catalog construction."""

from __future__ import annotations

from design_events.build_events.probability.dependence import (
    DriverDependenceModel,
    check_stress_budget,
    fit_driver_dependence,
    sample_tail_enriched_catalog,
)
from design_events.build_events.probability.design_catalog import (
    JointCatalogResult,
    build_tail,
    build_joint_catalog,
    fit_index_marginal,
)
from design_events.build_events.probability.inland_dependence import (
    InlandDesignCatalogResult,
    build_inland_catalog,
    build_inland_historical_tail_catalog,
    fit_reference_streamflow_pot,
)
from design_events.build_events.probability.exceedance import (
    AndExceedanceLabels,
    and_joint_survival,
    and_label_frame,
    and_return_period,
    label_and_joint_exceedance,
    select_most_likely_design_events,
)
from design_events.build_events.probability.realization import (
    attach_field_preserving_realization,
    draw_relative_lags,
    select_analog_realization,
)


__all__ = [
    "AndExceedanceLabels",
    "DriverDependenceModel",
    "InlandDesignCatalogResult",
    "JointCatalogResult",
    "and_joint_survival",
    "and_label_frame",
    "and_return_period",
    "attach_field_preserving_realization",
    "build_inland_catalog",
    "build_inland_historical_tail_catalog",
    "build_joint_catalog",
    "build_tail",
    "check_stress_budget",
    "draw_relative_lags",
    "fit_driver_dependence",
    "fit_index_marginal",
    "fit_reference_streamflow_pot",
    "label_and_joint_exceedance",
    "sample_tail_enriched_catalog",
    "select_analog_realization",
    "select_most_likely_design_events",
]
