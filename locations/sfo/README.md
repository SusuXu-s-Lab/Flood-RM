# San Francisco YAML Map

This folder is a placeholder location. It does not currently have a configured `config.yaml`.

When it becomes configured, follow the same pattern:

- `config.yaml` - location identity, flood mode, event drivers, scenarios, and included YAML files.
- `smartds.yaml` or `grid.yaml` - region/grid data pointers and AOI inputs.
- `sfincs.yaml` - SFINCS settings, forcing, parameters, and HydroMT-SFINCS recipes.
- `snapwave.yaml` - only if the location uses coastal wave coupling.
- `wflow.yaml` - only if the location uses Wflow coupling.

Do not check in `config.resolved.yaml`; generate merged views only when needed.
