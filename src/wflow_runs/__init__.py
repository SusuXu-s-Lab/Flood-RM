import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from wflow_runs.types import WflowBuildPlan, WflowDomainSetPlan, WflowSourceStrategy
from wflow_runs.qa import event_peak_discharge_table
from wflow_runs.river_geometry import validate_geometry
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
    "read_submodel_gauge_discharge": "wflow_runs.output",
    "replay_inland_domain_set": "wflow_runs.replay",
    "run_zero_rain_control": "wflow_runs.replay",
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
    "event_peak_discharge_table",
    "read_submodel_gauge_discharge",
    "replay_inland_domain_set",
    "run_zero_rain_control",
    "validate_geometry",
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
]
