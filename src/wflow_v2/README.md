# wflow_boundary native refactor

A minimal HydroMT-Wflow docs/runtime component for stochastic Wflow-to-SFINCS discharge boundary conditions.

The scientific package is `wflow_boundary`. Compatibility and project-specific glue lives outside the core package in `wflow_boundary_compat`.

```python
from wflow_boundary import plan_domain, build_base_models, prepare_states, run_event_boundary

submodels = plan_domain(config, location_root)
build_report = build_base_models(config, location_root)
state_report = prepare_states(config, location_root)
run = run_event_boundary(config, location_root, event_id="E0001", execute=True)
```

## Scientific contract

For each stochastic event `omega`, Wflow generates the rainfall-runoff response at reviewed SFINCS handoff points:

```text
Q^W_omega(h,t) = W_theta(P_omega,T_omega,PET_omega,S_0,omega)_h
```

The SFINCS forcing is written as:

```text
data/wflow/events/<event_id>/sfincs_discharge.nc
```

with variables and coordinates:

```text
discharge(index, time)
name(index)
x(index)
y(index)
```

Optional same-frequency amplification is a single documented scalar:

```text
K_omega = clip(Q*_omega(g*) / max_t Q^W_omega(g*,t), K_min, K_max)
```

## Native HydroMT-Wflow boundary

The core package now delegates the following to HydroMT-Wflow or the thin compatibility layer:

- `WflowSbmModel.build(steps=...)` for base model construction.
- `WflowSbmModel.update(...)` for event forcing model updates.
- Native workflow YAML reading through `hydromt.readers.read_workflow_yaml` when available.
- Native `setup_gauges` workflow steps for SFINCS and observation gauge layers.
- Native `setup_cold_states`, `model.states.write`, and `model.config.write` for instates.
- Native `hydromt_wflow.utils.read_csv_output` for Wflow CSV output parsing, with a fallback parser outside the core package.

## Compatibility boundary

`wflow_boundary_compat` contains shims for local catalogs, signature changes, output parsing fallback, runner commands, and removed legacy repair markers. It is import-connected but not part of the scientific implementation.

Project-specific migration repairs should live in a separate package or script that runs before `wflow_boundary`.
