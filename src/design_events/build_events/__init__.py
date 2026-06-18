"""Build sampled peaks, event catalogs, and surge hydrographs."""

from design_events.build_events.plan import (
    EventCatalogPlan,
    EventForcingPlan,
    build_event_catalog_plan,
)
from design_events.build_events.inland_event_catalog import (
    InlandEventArtifacts,
    build_inland_event_artifacts,
    write_wflow_sfincs_handoff_manifest,
)
from design_events.build_events.inland_streamflow import build_usgs_streamflow_event_members
from design_events.build_events.dependence import (
    DriverDependenceModel,
    check_stress_budget,
    fit_driver_dependence,
    sample_tail_enriched_catalog,
)
from design_events.build_events.joint_exceedance import (
    AndExceedanceLabels,
    and_joint_survival,
    and_label_frame,
    and_return_period,
    label_and_joint_exceedance,
    select_most_likely_design_events,
)
from design_events.build_events.joint_catalog import (
    JointCatalogResult,
    build_historical_tail_catalog,
    build_joint_design_catalog,
    fit_index_marginal,
)
from design_events.build_events.realization import (
    attach_field_preserving_realization,
    draw_relative_lags,
    select_analog_realization,
)

__all__ = [
    "AndExceedanceLabels",
    "DriverDependenceModel",
    "EventCatalogPlan",
    "EventForcingPlan",
    "InlandEventArtifacts",
    "JointCatalogResult",
    "and_joint_survival",
    "and_label_frame",
    "and_return_period",
    "attach_field_preserving_realization",
    "build_event_catalog_plan",
    "build_historical_tail_catalog",
    "build_inland_event_artifacts",
    "build_joint_design_catalog",
    "build_usgs_streamflow_event_members",
    "check_stress_budget",
    "draw_relative_lags",
    "fit_driver_dependence",
    "fit_index_marginal",
    "label_and_joint_exceedance",
    "sample_tail_enriched_catalog",
    "select_analog_realization",
    "select_most_likely_design_events",
    "write_wflow_sfincs_handoff_manifest",
]
