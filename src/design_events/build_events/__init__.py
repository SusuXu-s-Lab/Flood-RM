"""Build sampled peaks, event catalogs, and surge hydrographs."""

from design_events.build_events.workflow import (
    EventCatalogPlan,
    EventForcingPlan,
    plan,
)
from design_events.build_events.inland import (
    InlandEventArtifacts,
    build_inland_event_artifacts,
    write_handoff,
)
from design_events.build_events.inland import build_usgs_streamflow_event_members
from design_events.build_events.probability import (
    JointCatalogResult,
    and_return_period,
    attach_field_preserving_realization,
    build_tail,
    build_joint_catalog,
    check_stress_budget,
    draw_relative_lags,
    fit_index_marginal,
    select_analog_realization,
    select_most_likely_design_events,
)

__all__ = [
    "EventCatalogPlan",
    "EventForcingPlan",
    "InlandEventArtifacts",
    "JointCatalogResult",
    "and_return_period",
    "attach_field_preserving_realization",
    "plan",
    "build_tail",
    "build_inland_event_artifacts",
    "build_joint_catalog",
    "build_usgs_streamflow_event_members",
    "check_stress_budget",
    "draw_relative_lags",
    "fit_index_marginal",
    "select_analog_realization",
    "select_most_likely_design_events",
    "write_handoff",
]
