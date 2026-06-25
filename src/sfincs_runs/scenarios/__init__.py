from sfincs_runs.scenarios.audit import (
    ForcingAuditIssue,
    ForcingManifestAudit,
    audit_forcing_manifest,
)
from sfincs_runs.scenarios.coastal_realization import (
    build_coastal_event_timeseries,
    build_coastal_hydrograph_from_analog,
)
from sfincs_runs.scenarios.inland_coupled import (
    InlandCoupledExamplePlan,
    accepted_dynamic_handoff_event_ids,
    audit_inland_coupled_batch_readiness,
    dynamic_handoff_readiness_table,
    plan_inland_coupled_example,
    stage_inland_coupled_scenarios,
    stage_inland_coupled_scenario_forcing,
)
from sfincs_runs.scenarios.inland_initial_conditions import (
    configure_hydrograph_initial_conditions,
    derive_hydrograph_initial_depth,
)
from sfincs_runs.scenarios.joint_handoff import write_joint_catalog_sfincs_handoff

audit_forcing = audit_forcing_manifest
build_timeseries = build_coastal_event_timeseries
handoff_readiness = dynamic_handoff_readiness_table
init_hydrographs = configure_hydrograph_initial_conditions
plan_example = plan_inland_coupled_example
stage_scenarios = stage_inland_coupled_scenarios
write_handoff = write_joint_catalog_sfincs_handoff

__all__ = [
    "ForcingAuditIssue",
    "ForcingManifestAudit",
    "InlandCoupledExamplePlan",
    "accepted_dynamic_handoff_event_ids",
    "audit_forcing_manifest",
    "audit_inland_coupled_batch_readiness",
    "build_coastal_event_timeseries",
    "build_coastal_hydrograph_from_analog",
    "configure_hydrograph_initial_conditions",
    "derive_hydrograph_initial_depth",
    "dynamic_handoff_readiness_table",
    "plan_inland_coupled_example",
    "stage_inland_coupled_scenarios",
    "stage_inland_coupled_scenario_forcing",
    "write_joint_catalog_sfincs_handoff",
    "audit_forcing",
    "build_timeseries",
    "handoff_readiness",
    "init_hydrographs",
    "plan_example",
    "stage_scenarios",
    "write_handoff",
]
