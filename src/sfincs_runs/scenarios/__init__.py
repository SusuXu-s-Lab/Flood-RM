from sfincs_runs.scenarios.audit import (
    ForcingAuditIssue,
    ForcingManifestAudit,
    audit_forcing,
)
from sfincs_v2.coastal import build_coastal_hydrograph_from_analog as _v2_build_coastal_hydrograph_from_analog
from sfincs_v2.coastal import coastal_timeseries_from_catalog_row as build_timeseries
from sfincs_runs.scenarios.inland_coupled import (
    InlandCoupledForcingStage,
    InlandCoupledExamplePlan,
    accepted_dynamic_handoff_event_ids,
    audit_inland_coupled_batch_readiness,
    handoff_readiness,
    plan_example,
    stage_inland_coupled_example_forcing,
    stage_scenarios,
    stage_inland_coupled_scenario_forcing,
)
from sfincs_runs.scenarios.inland_initial_conditions import (
    init_hydrographs,
    derive_hydrograph_initial_depth,
)
from sfincs_runs.scenarios.joint_handoff import write_handoff
from sfincs_runs.scenarios.outcome_catalogue import (
    FloodOutcomeCatalogue,
    build_flood_event_outcome_catalogue,
)


def build_coastal_hydrograph_from_analog(
    components,
    peak_time,
    scale_factor,
    *,
    window_hours=72.0,
    msl_offset_m=0.0,
):
    """Compatibility wrapper for legacy relative-hour coastal realizations."""
    return _v2_build_coastal_hydrograph_from_analog(
        components,
        peak_time,
        scale_factor,
        window_hours=window_hours,
        msl_offset_m=msl_offset_m,
        return_absolute_time=False,
    )


__all__ = [
    "ForcingAuditIssue",
    "ForcingManifestAudit",
    "FloodOutcomeCatalogue",
    "InlandCoupledExamplePlan",
    "InlandCoupledForcingStage",
    "accepted_dynamic_handoff_event_ids",
    "audit_forcing",
    "audit_inland_coupled_batch_readiness",
    "build_timeseries",
    "build_coastal_hydrograph_from_analog",
    "build_flood_event_outcome_catalogue",
    "init_hydrographs",
    "derive_hydrograph_initial_depth",
    "handoff_readiness",
    "plan_example",
    "stage_inland_coupled_example_forcing",
    "stage_scenarios",
    "stage_inland_coupled_scenario_forcing",
    "write_handoff",
    "audit_forcing",
    "build_timeseries",
    "handoff_readiness",
    "init_hydrographs",
    "plan_example",
    "stage_scenarios",
    "write_handoff",
]
