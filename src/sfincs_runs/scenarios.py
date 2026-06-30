import sys

from sfincs_runs import audit as audit
from sfincs_runs import create_events as create_events
from sfincs_runs import event_forcing as event_forcing
from sfincs_runs import inland_coupled as inland_coupled
from sfincs_runs import inland_initial_conditions as inland_initial_conditions
from sfincs_runs import joint_handoff as joint_handoff
from sfincs_runs import outcome_catalogue as outcome_catalogue
from sfincs_runs import run_events as run_events
from sfincs_runs import run_inland_coupled_events as run_inland_coupled_events
from sfincs_runs import scenario_events as scenarios
from sfincs_runs import scenario_stats as scenario_stats
from sfincs_runs import timing as timing
from sfincs_runs.audit import (
    ForcingAuditIssue,
    ForcingManifestAudit,
    audit_forcing,
)
from sfincs_runs.coastal import build_coastal_hydrograph_from_analog as _v2_build_coastal_hydrograph_from_analog
from sfincs_runs.coastal import coastal_timeseries_from_catalog_row as build_timeseries
from sfincs_runs.inland_coupled import (
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
from sfincs_runs.inland_initial_conditions import (
    init_hydrographs,
    derive_hydrograph_initial_depth,
)
from sfincs_runs.joint_handoff import write_handoff
from sfincs_runs.outcome_catalogue import (
    FloodOutcomeCatalogue,
    build_flood_event_outcome_catalogue,
)

__path__ = []
if __spec__ is not None:
    __spec__.submodule_search_locations = __path__

for _name, _module in {
    "audit": audit,
    "create_events": create_events,
    "event_forcing": event_forcing,
    "inland_coupled": inland_coupled,
    "inland_initial_conditions": inland_initial_conditions,
    "joint_handoff": joint_handoff,
    "outcome_catalogue": outcome_catalogue,
    "run_events": run_events,
    "run_inland_coupled_events": run_inland_coupled_events,
    "scenario_stats": scenario_stats,
    "scenarios": scenarios,
    "timing": timing,
}.items():
    sys.modules[f"{__name__}.{_name}"] = _module


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
