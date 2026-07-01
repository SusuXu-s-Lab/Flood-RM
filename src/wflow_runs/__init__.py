import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from wflow_runs.build_plan import (
    build_wflow_data_catalog,
    build_wflow_submodel,
    build_wflow_steps_for_submodel,
    build_wflow_build_plan,
    plan_wflow_us_source_strategy,
    plan_wflow_domain_set,
    plan_wflow_domain_set_from_streamgages,
    plan_wflow_domain_set_from_stream_boundary_crossings,
    plan_wflow_domain_set_from_boundary_handoff_watersheds,
    plan_wflow_domain_set_from_encompassing_huc,
    wflow_catalog_source_readiness,
    write_wflow_crossing_gauge_locations,
    write_wflow_sfincs_gauge_locations,
    write_wflow_observation_gauge_locations,
    write_wflow_subbasin_fabric_from_nhdplus,
    write_wflow_domain_set_manifest,
)
from wflow_runs.types import WflowBuildPlan, WflowDomainSetPlan, WflowSourceStrategy
from wflow_runs.qa import event_peak_discharge_table
from wflow_runs.river_geometry import validate_geometry
from wflow_runs.coupling_qa import (
    CoupledDomainReview,
    WflowArtifactInventory,
    WflowHandoffContract,
    coupled_domain_review,
    wflow_artifact_inventory,
    wflow_handoff_contract,
)
from wflow_runs.states import (
    configure_wflow_state_paths,
    plan_warmup,
    plan_wflow_warmup_state,
    prepare_instates,
    prepare_wflow_event_instate,
    promote_outstate_to_instate,
    shared_baseline_warmup_settings,
    validate_warmup_forcing,
    validate_wflow_reservoir_states,
    validate_instates,
    warmup_window,
    write_cold_state_workflow,
)
_LAZY_EXPORTS = {
    "build_meteo": "wflow_runs.replay",
    "read_submodel_gauge_discharge": "wflow_runs.output",
    "replay_inland_domain_set": "wflow_runs.replay",
    "resolve_event_rainfall_source_nc": "wflow_runs.replay",
    "resolve_event_window": "wflow_runs.replay",
    "run_zero_rain_control": "wflow_runs.replay",
    "prepare_wflow_streamflow_realization_for_event_model": "wflow_runs.streamflow_realization",
    "validate_wflow_streamflow_realization": "wflow_runs.streamflow_realization",
    "wflow_streamflow_gage_overlap": "wflow_runs.streamflow_realization",
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(_LAZY_EXPORTS[name]), name)
    globals()[name] = value
    return value


__all__ = [
    "WflowBuildPlan",
    "WflowDomainSetPlan",
    "WflowSourceStrategy",
    "build_wflow_data_catalog",
    "build_wflow_submodel",
    "build_wflow_steps_for_submodel",
    "build_wflow_build_plan",
    "plan_wflow_us_source_strategy",
    "plan_wflow_domain_set",
    "plan_wflow_domain_set_from_streamgages",
    "plan_wflow_domain_set_from_stream_boundary_crossings",
    "plan_wflow_domain_set_from_boundary_handoff_watersheds",
    "plan_wflow_domain_set_from_encompassing_huc",
    "wflow_catalog_source_readiness",
    "write_wflow_crossing_gauge_locations",
    "write_wflow_sfincs_gauge_locations",
    "write_wflow_observation_gauge_locations",
    "write_wflow_subbasin_fabric_from_nhdplus",
    "write_wflow_domain_set_manifest",
    "event_peak_discharge_table",
    "resolve_event_window",
    "resolve_event_rainfall_source_nc",
    "build_meteo",
    "read_submodel_gauge_discharge",
    "replay_inland_domain_set",
    "run_zero_rain_control",
    "validate_geometry",
    "CoupledDomainReview",
    "WflowArtifactInventory",
    "WflowHandoffContract",
    "coupled_domain_review",
    "wflow_artifact_inventory",
    "wflow_handoff_contract",
    "warmup_window",
    "write_cold_state_workflow",
    "configure_wflow_state_paths",
    "prepare_instates",
    "promote_outstate_to_instate",
    "prepare_wflow_event_instate",
    "plan_warmup",
    "plan_wflow_warmup_state",
    "shared_baseline_warmup_settings",
    "validate_warmup_forcing",
    "validate_wflow_reservoir_states",
    "validate_instates",
    "prepare_wflow_streamflow_realization_for_event_model",
    "validate_wflow_streamflow_realization",
    "wflow_streamflow_gage_overlap",
]
