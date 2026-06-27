# Model YAML Source of Truth

## Status

Accepted.

## Context

ADR-0014 made HydroMT build/update recipes visible by placing separate
`sfincs_build.yml`, `sfincs_update_forcing.yml`, `wflow_build.yml`, and
`wflow_update_forcing.yml` files at each Location Workspace root. That improved
auditability, but it split one model's settings, forcing, parameters, and
HydroMT recipe across multiple root files.

Stakeholders now need one clear YAML per model while still preserving native
HydroMT recipe syntax.

## Decision

Use one human-authored YAML file per model:

- `sfincs.yaml`
- `wflow.yaml`
- `snapwave.yaml`

Each model YAML owns model settings, forcing, parameters, and native HydroMT
recipes under `hydromt:`. Wflow recipes keep the native HydroMT `steps:` list.
SFINCS and SnapWave recipes keep native HydroMT-SFINCS setup/update step names.

`config.yaml` remains a small launch file with identity, flood setting, event
drivers, user-facing scenarios, and `includes:` only. SMART-DS region pointers
live in `smartds.yaml`; Marshfield keeps `grid.yaml` because its grid is
generated, not a SMART-DS dataset.

Generated native HydroMT recipe files may be materialized under
`data/<model>/config/` for CLI calls, but they are artifacts, not source config.

## Consequences

- This supersedes ADR-0014's root-level recipe-file layout while preserving its
  native-HydroMT-syntax requirement.
- Root-level `config.resolved.yaml` files are removed from Location Workspaces;
  merged views can still be generated on demand.
- Generated YAML artifacts need visible generated-file notices so readers do not
  confuse them with hand-authored model configuration.
